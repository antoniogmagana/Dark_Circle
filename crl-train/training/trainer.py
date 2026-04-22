"""CRLModel assembly and Trainer for CRL pre-training and downstream evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import IO, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.models.frontend import MultiScale1DFrontend, MorletFilterbank
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder
from crl_vehicle.models.intervention import UnknownInterventionClassifier
from crl_vehicle.models.heads import (
    LinearPresenceHead, LinearTypeHead, LinearProximityHead,
    MLPTypeHead, FullZTypeHead,
)
from crl_vehicle.training_modes import (
    CheckpointState, TrainingMode, build_training_mode,
)


# ---------------------------------------------------------------------------
# CRLModel
# ---------------------------------------------------------------------------

class CRLModel(nn.Module):
    """Full CRL model supporting multiscale (early fusion) and morlet (late fusion) frontends.

    Multiscale path:
        per-sensor frontends (MultiScale1D + AvgPool + AdaptiveAvgPool) →
        time-concat → single shared TemporalEncoder → shared FeatureDecoder

    Morlet path:
        per-sensor frontends (MorletFilterbank + AvgPool) →
        per-sensor TemporalEncoder → per-sensor FeatureDecoder
    """

    VALID_PROBE_MODES = ("linear_ztype", "mlp_ztype", "linear_fullz")

    def __init__(
        self,
        config: CRLConfig,
        sensors: list[str] | None = None,
        probe_mode: str = "linear_ztype",
    ) -> None:
        super().__init__()
        self.cfg = config
        self.sensors = sensors or ["audio", "seismic"]
        if probe_mode not in self.VALID_PROBE_MODES:
            raise ValueError(
                f"probe_mode must be one of {self.VALID_PROBE_MODES}, got {probe_mode!r}"
            )
        self.probe_mode = probe_mode
        d_z = config.d_z

        self.frontends = nn.ModuleDict()
        self.latent = CausalLatentSpace(d_z=d_z)
        self.interv_classifier = UnknownInterventionClassifier()

        if config.frontend_type == "multiscale":
            self._init_multiscale(config, d_z)
            head_keys = ["fused"]
        elif config.frontend_type == "morlet":
            self._init_morlet(config, d_z)
            head_keys = self.sensors
        elif config.frontend_type == "morlet_per_sensor":
            self._init_morlet_per_sensor(config, d_z)
            head_keys = self.sensors
        else:
            raise ValueError(f"Unknown frontend_type: {config.frontend_type!r}")

        # Downstream heads (one set per head_key)
        self.pres_heads = nn.ModuleDict()
        self.type_heads = nn.ModuleDict()
        self.prox_heads = nn.ModuleDict()
        # Aux heads for supervised signal during CRL pre-training
        self.aux_pres_heads = nn.ModuleDict()
        self.aux_type_heads = nn.ModuleDict()

        for key in head_keys:
            self.pres_heads[key]     = LinearPresenceHead()
            self.type_heads[key]     = self._build_type_head(d_z)
            self.prox_heads[key]     = LinearProximityHead()
            self.aux_pres_heads[key] = nn.Linear(CausalLatentSpace.D_PRES, 1)
            self.aux_type_heads[key] = nn.Linear(CausalLatentSpace.D_TYPE, 4)

    def _build_type_head(self, d_z: int) -> nn.Module:
        if self.probe_mode == "linear_ztype":
            return LinearTypeHead()
        if self.probe_mode == "mlp_ztype":
            return MLPTypeHead()
        if self.probe_mode == "linear_fullz":
            return FullZTypeHead(d_z=d_z)
        raise ValueError(f"Unknown probe_mode: {self.probe_mode!r}")

    def _init_multiscale(self, config: CRLConfig, d_z: int) -> None:
        T = config.fused_seq_len
        stride = config.multiscale_pool_stride
        for sensor in self.sensors:
            mc = config.modality_cfg(sensor)
            self.frontends[sensor] = nn.Sequential(
                MultiScale1DFrontend(mc.n_channels, config.d_model),
                nn.AvgPool1d(stride, stride),
                nn.AdaptiveAvgPool1d(T),
            )
        self.encoder = TemporalEncoder(
            in_channels=config.d_model,
            d_z=d_z,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
        self.decoder = FeatureDecoder(
            out_channels=config.d_model,
            seq_len=len(self.sensors) * T,
            d_z=d_z,
            d_model=config.d_model,
        )
        # Empty per-sensor dicts for structural consistency
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

    def _init_morlet(self, config: CRLConfig, d_z: int) -> None:
        self.encoder = None
        self.decoder = None
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        stride = config.morlet_pool_stride
        use_phase = getattr(config, "morlet_use_phase", False)
        for sensor in self.sensors:
            mc = config.modality_cfg(sensor)
            ks = config.morlet_kernel_size
            kernel_mb = (2 * config.d_model * mc.n_channels * ks * 4) / 1e6
            print(f"  MorletFilterbank [{sensor}]: kernel {2*config.d_model}×{mc.n_channels}×{ks} = {kernel_mb:.2f} MB")
            bank = MorletFilterbank(
                mc.n_channels, config.d_model, ks, mc.sample_rate,
                use_phase=use_phase,
            )
            self.frontends[sensor] = nn.Sequential(
                bank, nn.AvgPool1d(stride, stride),
            )
            # Downstream encoder sees bank.total_out_channels (3× if phase).
            enc_in_channels = bank.total_out_channels
            seq_len = mc.window_size // stride
            self.encoders[sensor] = TemporalEncoder(
                in_channels=enc_in_channels,
                d_z=d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            # Decoder must match frontend channel count — recon target is
            # the frontend's raw output, which is 3× when use_phase=True.
            self.decoders[sensor] = FeatureDecoder(
                out_channels=enc_in_channels,
                seq_len=max(1, seq_len),
                d_z=d_z,
                d_model=config.d_model,
            )

    def _init_morlet_per_sensor(self, config: CRLConfig, d_z: int) -> None:
        """Per-sensor Morlet banks with sensor-specific freq ranges.

        Audio and seismic each get a bank whose freq_min/freq_max, channel
        count, and w0 come from config.morlet_per_sensor_params[sensor].
        Otherwise identical plumbing to _init_morlet: late fusion, one
        encoder/decoder per sensor.
        """
        self.encoder = None
        self.decoder = None
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        stride = config.morlet_pool_stride
        use_phase = config.morlet_use_phase
        params_by_sensor = config.morlet_per_sensor_params

        for sensor in self.sensors:
            if sensor not in params_by_sensor:
                raise ValueError(
                    f"morlet_per_sensor requires params for {sensor!r} in "
                    f"config.morlet_per_sensor_params (got keys "
                    f"{list(params_by_sensor.keys())})"
                )
            sp = params_by_sensor[sensor]
            mc = config.modality_cfg(sensor)
            ks = config.morlet_kernel_size
            out_channels = max(1, int(round(config.d_model * sp.get("out_channels_frac", 1.0))))
            freq_min = float(sp["freq_min"])
            freq_max = float(sp["freq_max"])
            w0 = float(sp.get("w0", 6.0))

            kernel_mb = (2 * out_channels * mc.n_channels * ks * 4) / 1e6
            print(
                f"  MorletFilterbank [{sensor}]: "
                f"{out_channels}ch × [{freq_min:g}, {freq_max:g}] Hz, "
                f"w0={w0}, ks={ks} = {kernel_mb:.2f} MB"
                + (", +phase" if use_phase else "")
            )

            bank = MorletFilterbank(
                in_channels=mc.n_channels,
                out_channels=out_channels,
                kernel_size=ks,
                sample_rate=mc.sample_rate,
                w0=w0,
                freq_min=freq_min,
                freq_max=freq_max,
                use_phase=use_phase,
            )
            self.frontends[sensor] = nn.Sequential(
                bank, nn.AvgPool1d(stride, stride),
            )
            enc_in_channels = bank.total_out_channels
            seq_len = mc.window_size // stride
            self.encoders[sensor] = TemporalEncoder(
                in_channels=enc_in_channels,
                d_z=d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            self.decoders[sensor] = FeatureDecoder(
                out_channels=enc_in_channels,
                seq_len=max(1, seq_len),
                d_z=d_z,
                d_model=config.d_model,
            )

    # ------------------------------------------------------------------
    # Encode / decode API
    # ------------------------------------------------------------------

    def encode_fused(
        self, x_audio: torch.Tensor, x_seismic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Early-fusion encode. Returns (features, z, mu, logvar).
        features: (B, d_model, n_sensors*fused_seq_len) — reconstruction target."""
        feats = [self.frontends[s](x.float())
                 for s, x in zip(self.sensors, [x_audio, x_seismic])]
        features = torch.cat(feats, dim=2)
        z, mu, logvar = self.encoder(features)
        return features, z, mu, logvar

    def decode_fused(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode(
        self, sensor: str, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sensor encode (morlet path). Returns (features, z, mu, logvar)."""
        features = self.frontends[sensor](x.float())
        z, mu, logvar = self.encoders[sensor](features)
        return features, z, mu, logvar

    def decode(self, sensor: str, z: torch.Tensor) -> torch.Tensor:
        return self.decoders[sensor](z)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def backbone_parameters(self) -> list[nn.Parameter]:
        exclude_ids = set(
            id(p)
            for group in [
                self.pres_heads, self.type_heads, self.prox_heads,
                self.aux_pres_heads, self.aux_type_heads,
            ]
            for p in group.parameters()
        )
        return [p for p in self.parameters() if id(p) not in exclude_ids]

    def head_parameters(self) -> Iterator[nn.Parameter]:
        for group in [self.pres_heads, self.type_heads, self.prox_heads]:
            yield from group.parameters()

    def _finetune_params(self, top_n: int) -> list[nn.Parameter]:
        """Return backbone parameters to unfreeze for fine-tuning.

        top_n == -1: entire backbone.
        top_n >= 1:  top N transformer layers + mu/lv projection heads.
        """
        if top_n == -1:
            return list(self.backbone_parameters())

        params: list[nn.Parameter] = []
        if self.cfg.frontend_type == "multiscale":
            enc = self.encoder
            layers = list(enc.transformer.layers)
            for layer in layers[-top_n:]:
                params += list(layer.parameters())
            params += list(enc.mu_head.parameters())
            params += list(enc.lv_head.parameters())
        else:
            for enc in self.encoders.values():
                layers = list(enc.transformer.layers)
                for layer in layers[-top_n:]:
                    params += list(layer.parameters())
                params += list(enc.mu_head.parameters())
                params += list(enc.lv_head.parameters())
        return params


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _binary_f1_acc(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Binary F1 and accuracy from raw logits and {0,1} labels."""
    if logits.numel() == 0:
        return 0.0, 0.0
    preds = (logits > 0).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    f1  = (2 * tp) / max(2 * tp + fp + fn, 1)
    acc = (preds == labels).float().mean().item()
    return f1, acc


def _macro_f1_acc(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> tuple[float, float]:
    """Macro-averaged F1 and accuracy from class logits and integer labels."""
    if logits.numel() == 0:
        return 0.0, 0.0
    preds = logits.argmax(dim=-1)
    acc   = (preds == labels).float().mean().item()
    f1_sum = 0.0
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        f1_sum += (2 * tp) / max(2 * tp + fp + fn, 1)
    return f1_sum / n_classes, acc


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Handles CRL pre-training and downstream head training.

    Owns the epoch loop, optimizer, CSV logging, and patience counter. All
    algorithm-specific logic (loss computation, beta schedule, checkpoint
    selection) lives in a TrainingMode instance built from config.
    """

    def __init__(
        self,
        model: CRLModel,
        config: CRLConfig,
        device: torch.device,
        save_dir: Path,
    ) -> None:
        self.model  = model
        self.cfg    = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.mode: TrainingMode = build_training_mode(config).to(device)
        self.ckpt_state = CheckpointState()

        self.beta = 0.0
        self._pres_pos_weight:    torch.Tensor | None = None
        self._type_class_weights: torch.Tensor | None = None

        # The optimizer covers both model backbone and any learnable params
        # carried by the mode (e.g. ConditionalPrior MLP at Checkpoint 2).
        backbone_param_ids = {id(p) for p in model.backbone_parameters()}
        mode_params = [p for p in self.mode.parameters()
                       if id(p) not in backbone_param_ids]
        param_groups = [
            {"params": model.backbone_parameters(), "weight_decay": config.wd}
        ]
        if mode_params:
            param_groups.append({"params": mode_params, "weight_decay": config.wd})
        self.optimizer = torch.optim.AdamW(param_groups, lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.n_epochs, eta_min=config.lr_min
        )

    # ------------------------------------------------------------------
    # Forward pair — delegates to the TrainingMode.
    # ------------------------------------------------------------------

    def _forward_pair(
        self, batch: dict, beta: float,
    ) -> tuple[torch.Tensor, dict]:
        return self.mode.forward_pair(self.model, batch, beta, self.device)

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    _AUX_TENSOR_KEYS = (
        "aux_pres_logits", "aux_pres_labels",
        "aux_type_logits", "aux_type_labels",
    )

    def _accumulate(
        self,
        scalar_agg: dict[str, float],
        tensor_agg: dict[str, list[torch.Tensor]],
        metrics: dict,
    ) -> None:
        for k, v in metrics.items():
            if k in self._AUX_TENSOR_KEYS:
                tensor_agg.setdefault(k, []).append(v)
            else:
                scalar_agg[k] = scalar_agg.get(k, 0.0) + v

    def _finalize_epoch_metrics(
        self,
        scalar_agg: dict[str, float],
        tensor_agg: dict[str, list[torch.Tensor]],
        n_batches: int,
        prefix: str,
    ) -> dict:
        out = {f"{prefix}{k}": v / max(n_batches, 1) for k, v in scalar_agg.items()}
        pres_logits = torch.cat(tensor_agg.get("aux_pres_logits", [])) \
            if tensor_agg.get("aux_pres_logits") else torch.empty(0)
        pres_labels = torch.cat(tensor_agg.get("aux_pres_labels", [])) \
            if tensor_agg.get("aux_pres_labels") else torch.empty(0, dtype=torch.long)
        type_logits = torch.cat(tensor_agg.get("aux_type_logits", [])) \
            if tensor_agg.get("aux_type_logits") else torch.empty(0)
        type_labels = torch.cat(tensor_agg.get("aux_type_labels", [])) \
            if tensor_agg.get("aux_type_labels") else torch.empty(0, dtype=torch.long)
        pres_f1, pres_acc = _binary_f1_acc(pres_logits, pres_labels)
        type_f1, type_acc = _macro_f1_acc(type_logits, type_labels, n_classes=4)
        out[f"{prefix}aux_pres_f1"]  = round(pres_f1,  4)
        out[f"{prefix}aux_pres_acc"] = round(pres_acc, 4)
        out[f"{prefix}aux_type_f1"]  = round(type_f1,  4)
        out[f"{prefix}aux_type_acc"] = round(type_acc, 4)
        return out

    def _train_epoch(
        self, loader: DataLoader, beta: float, steps_per_epoch: int | None = None
    ) -> dict:
        self.model.train()
        scalar_agg: dict[str, float] = {}
        tensor_agg: dict[str, list[torch.Tensor]] = {}
        n = 0
        for i, batch in enumerate(loader):
            if steps_per_epoch is not None and i >= steps_per_epoch:
                break
            self.optimizer.zero_grad()
            loss, metrics = self._forward_pair(batch, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._accumulate(scalar_agg, tensor_agg, metrics)
            n += 1
        return self._finalize_epoch_metrics(scalar_agg, tensor_agg, n, prefix="train_")

    def _eval_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.eval()
        scalar_agg: dict[str, float] = {}
        tensor_agg: dict[str, list[torch.Tensor]] = {}
        n = 0
        with torch.no_grad():
            for batch in loader:
                _, metrics = self._forward_pair(batch, beta=1.0)
                self._accumulate(scalar_agg, tensor_agg, metrics)
                n += 1
        out = self._finalize_epoch_metrics(scalar_agg, tensor_agg, n, prefix="val_")
        return self.mode.val_metrics_summary(out)

    # ------------------------------------------------------------------
    # CRL pre-training
    # ------------------------------------------------------------------

    def train_crl(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        steps_per_epoch: int | None = None,
    ) -> None:
        """Run CRL pre-training. Algorithm-specific logic (beta, checkpoints,
        early stopping metric) is delegated to self.mode; this method owns
        only the epoch loop, CSV logging, and patience counting."""
        early_stop_metric = self.mode.early_stop_metric()
        csv_path = self.save_dir / "crl_metrics.csv"
        csv_file: IO | None = None
        csv_writer: csv.DictWriter | None = None

        try:
            for epoch in range(epochs):
                train_m = self._train_epoch(train_loader, self.beta, steps_per_epoch)
                val_m   = self._eval_epoch(val_loader, self.beta)
                self.scheduler.step()

                new_beta, event = self.mode.update_beta(
                    self.beta, val_m, self.ckpt_state, self.cfg
                )
                self.beta = new_beta

                row = {"epoch": epoch, "beta": self.beta, "beta_event": event}
                row.update(train_m)
                row.update(val_m)
                if csv_writer is None:
                    csv_file = open(csv_path, "w", newline="")
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()

                print(
                    f"Epoch {epoch:3d} | beta={self.beta:.3f} {event} | "
                    f"recon={val_m.get('val_recon',0):.4f} "
                    f"kl={val_m.get('val_raw_kl',0):.4f} "
                    f"{early_stop_metric}={val_m.get(early_stop_metric, 0):.4f} "
                    f"aux_pres_f1={val_m.get('val_aux_pres_f1',0):.3f} "
                    f"aux_type_f1={val_m.get('val_aux_type_f1',0):.3f}"
                )

                saves = self.mode.should_save_checkpoint(val_m, epoch, self.ckpt_state)
                for ckpt_name, should_save in saves.items():
                    if should_save:
                        torch.save(self.model.state_dict(), self.save_dir / ckpt_name)

                if self.ckpt_state.patience_count >= self.cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        finally:
            if csv_file is not None:
                csv_file.close()

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")
        summary_path = self.save_dir / "crl_checkpoint_summary.json"
        summary_path.write_text(
            json.dumps(self.mode.checkpoint_summary(self.ckpt_state), indent=2)
        )

    # ------------------------------------------------------------------
    # Downstream training
    # ------------------------------------------------------------------

    def train_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        pres_pos_weight: torch.Tensor | None = None,
        type_class_weights: torch.Tensor | None = None,
        finetune_top_n: int = 0,
        ckpt_name: str = "crl_best.pth",
    ) -> None:
        """Train downstream heads on frozen (or partially unfrozen) CRL backbone.

        Args:
            pres_pos_weight:    BCEWithLogitsLoss pos_weight for presence head.
            type_class_weights: CrossEntropyLoss weight tensor for type head.
            finetune_top_n:     0  = fully frozen backbone (heads only).
                                N  = unfreeze top N transformer layers + mu/lv heads at 0.1x lr.
                                -1 = unfreeze entire backbone at 0.1x lr.
            ckpt_name:          Which CRL checkpoint to load as the frozen backbone.
                                Defaults to "crl_best.pth" (selected by val_ref_elbo).
                                Pass "crl_best_aux_type.pth" to use the F1-selected one.
        """
        best_path = self.save_dir / ckpt_name
        if best_path.exists():
            print(f"  Loading CRL backbone from {best_path.name}")
            state = torch.load(best_path, map_location=self.device)
            # Pre-filter: drop (aux_)type_heads tensors whose shape disagrees with
            # the current model. This happens when probe_mode differs from the
            # saved run (e.g., linear_ztype -> linear_fullz changes head in-features
            # from D_TYPE=6 to d_z=24). load_state_dict(strict=False) tolerates
            # missing/unexpected keys but NOT shape mismatches on shared keys, so
            # these must be removed here; they then surface as missing keys below.
            model_state = self.model.state_dict()
            dropped_shape_mismatch: list[str] = []
            for k in list(state.keys()):
                if (
                    k.startswith(("type_heads.", "aux_type_heads."))
                    and k in model_state
                    and state[k].shape != model_state[k].shape
                ):
                    dropped_shape_mismatch.append(k)
                    state.pop(k)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            # type_heads.* / aux_type_heads.* mismatches are expected when
            # --probe-mode differs from the saved run. Anything else indicates a
            # genuine shape/config mismatch (e.g., wrong d_z) and should fail loudly.
            def _is_type_head_key(k: str) -> bool:
                return k.startswith(("type_heads.", "aux_type_heads."))

            other_missing    = [k for k in missing    if not _is_type_head_key(k)]
            other_unexpected = [k for k in unexpected if not _is_type_head_key(k)]
            if other_missing or other_unexpected:
                raise RuntimeError(
                    f"Checkpoint load mismatch outside type_heads — refusing to "
                    f"silently train on partially-initialized backbone. "
                    f"Missing: {other_missing[:5]} | Unexpected: {other_unexpected[:5]}"
                )
            type_keys_missing    = [k for k in missing    if _is_type_head_key(k)]
            type_keys_unexpected = [k for k in unexpected if _is_type_head_key(k)]
            if type_keys_missing or type_keys_unexpected or dropped_shape_mismatch:
                print(
                    f"  Probe-mode head swap: {len(type_keys_missing)} new type-head "
                    f"param(s) will train from scratch; "
                    f"{len(type_keys_unexpected) + len(dropped_shape_mismatch)} "
                    f"old param(s) in checkpoint discarded."
                )

        # Freeze everything, then selectively unfreeze
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.head_parameters():
            p.requires_grad_(True)

        backbone_params: list[nn.Parameter] = []
        if finetune_top_n != 0:
            backbone_params = self._finetune_params(finetune_top_n)
            for p in backbone_params:
                p.requires_grad_(True)

        param_groups = [{"params": list(self.model.head_parameters()), "lr": self.cfg.lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.cfg.lr * 0.1})

        head_opt = torch.optim.AdamW(param_groups)
        self._pres_pos_weight    = pres_pos_weight
        self._type_class_weights = type_class_weights

        csv_path = self.save_dir / "downstream_metrics.csv"
        csv_file: IO | None = None
        csv_writer: csv.DictWriter | None = None
        best_val_loss = float("inf")

        try:
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                n = 0
                for batch in train_loader:
                    head_opt.zero_grad()
                    loss, _ = self._downstream_forward(batch)
                    loss.backward()
                    head_opt.step()
                    train_loss += loss.item()
                    n += 1

                self.model.eval()
                val_loss = 0.0
                m = 0
                pres_logits_all: list[torch.Tensor] = []
                pres_labels_all: list[torch.Tensor] = []
                type_logits_all: list[torch.Tensor] = []
                type_labels_all: list[torch.Tensor] = []
                with torch.no_grad():
                    for batch in val_loader:
                        loss, outputs = self._downstream_forward(batch)
                        val_loss += loss.item()
                        m += 1
                        pres_logits_all.append(outputs["pres_logits"].cpu())
                        pres_labels_all.append(outputs["pres_labels"].cpu())
                        if outputs["type_logits"] is not None:
                            type_logits_all.append(outputs["type_logits"].cpu())
                            type_labels_all.append(outputs["type_labels"].cpu())

                pres_f1, pres_acc = _binary_f1_acc(
                    torch.cat(pres_logits_all), torch.cat(pres_labels_all)
                )
                type_f1 = type_acc = 0.0
                if type_logits_all:
                    type_f1, type_acc = _macro_f1_acc(
                        torch.cat(type_logits_all), torch.cat(type_labels_all), n_classes=4
                    )

                avg_val = val_loss / max(m, 1)
                row = {
                    "epoch":        epoch,
                    "train_loss":   train_loss / max(n, 1),
                    "val_loss":     avg_val,
                    "val_pres_f1":  round(pres_f1,  4),
                    "val_pres_acc": round(pres_acc,  4),
                    "val_type_f1":  round(type_f1,  4),
                    "val_type_acc": round(type_acc,  4),
                }
                if csv_writer is None:
                    csv_file = open(csv_path, "w", newline="")
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(self.model.state_dict(), self.save_dir / "downstream_best.pth")

                print(
                    f"DS Epoch {epoch:3d} | train_loss={row['train_loss']:.4f} "
                    f"val_loss={avg_val:.4f} "
                    f"pres_f1={pres_f1:.3f} type_f1={type_f1:.3f}"
                )
        finally:
            if csv_file is not None:
                csv_file.close()
            self._pres_pos_weight    = None
            self._type_class_weights = None

        for p in self.model.parameters():
            p.requires_grad_(True)

    def _downstream_forward(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict]:
        """Return (loss, {pres_logits, pres_labels, type_logits, type_labels}).

        type_logits/type_labels are None when no valid-type samples exist in batch.
        Logits returned on the same device as model; accumulate on CPU for epoch metrics.
        """
        model = self.model
        dev   = self.device
        total = torch.tensor(0.0, device=dev)
        n = 0

        pres_logits_list: list[torch.Tensor] = []
        pres_labels_list: list[torch.Tensor] = []
        type_logits_list: list[torch.Tensor] = []
        type_labels_list: list[torch.Tensor] = []

        ppw = self._pres_pos_weight
        tcw = self._type_class_weights

        use_fullz = model.probe_mode == "linear_fullz"

        if self.cfg.frontend_type == "multiscale":
            avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
            if avail.any():
                x_a = batch["x_audio"][avail].to(dev)
                x_s = batch["x_seismic"][avail].to(dev)
                _, z, _, _ = model.encode_fused(x_a, x_s)
                z_pres, z_type, _, _, _ = model.latent.split(z)
                det   = batch["detection_label"][avail].float().to(dev)
                vtype = batch["vehicle_type"][avail].to(dev)
                pres_logit = model.pres_heads["fused"](z_pres).squeeze(-1)
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                    pres_logit, det, pos_weight=ppw)
                pres_logits_list.append(pres_logit.detach())
                pres_labels_list.append(det.long())
                valid = vtype >= 0
                if valid.any():
                    z_for_type = z[valid] if use_fullz else z_type[valid]
                    type_logit = model.type_heads["fused"](z_for_type)
                    total = total + torch.nn.functional.cross_entropy(
                        type_logit, vtype[valid], weight=tcw)
                    type_logits_list.append(type_logit.detach())
                    type_labels_list.append(vtype[valid])
                n += 1
        else:
            for sensor in model.sensors:
                avail = batch[f"{sensor}_avail"].bool()
                if not avail.any():
                    continue
                x = batch[f"x_{sensor}"][avail].to(dev)
                _, z, _, _ = model.encode(sensor, x)
                z_pres, z_type, _, _, _ = model.latent.split(z)
                det   = batch["detection_label"][avail].float().to(dev)
                vtype = batch["vehicle_type"][avail].to(dev)
                pres_logit = model.pres_heads[sensor](z_pres).squeeze(-1)
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                    pres_logit, det, pos_weight=ppw)
                pres_logits_list.append(pres_logit.detach())
                pres_labels_list.append(det.long())
                valid = vtype >= 0
                if valid.any():
                    z_for_type = z[valid] if use_fullz else z_type[valid]
                    type_logit = model.type_heads[sensor](z_for_type)
                    total = total + torch.nn.functional.cross_entropy(
                        type_logit, vtype[valid], weight=tcw)
                    type_logits_list.append(type_logit.detach())
                    type_labels_list.append(vtype[valid])
                n += 1

        outputs = {
            "pres_logits": torch.cat(pres_logits_list) if pres_logits_list else torch.empty(0, device=dev),
            "pres_labels": torch.cat(pres_labels_list) if pres_labels_list else torch.empty(0, dtype=torch.long, device=dev),
            "type_logits": torch.cat(type_logits_list) if type_logits_list else None,
            "type_labels": torch.cat(type_labels_list) if type_labels_list else None,
        }
        return total / max(n, 1), outputs

