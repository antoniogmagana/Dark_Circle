"""CRLModel assembly and Trainer for CRL pre-training and downstream evaluation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import IO, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.models.frontend import LearnableMorletFilterbank
from crl_vehicle.models.frontend_factory import build_frontend
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.intervention import UnknownInterventionClassifier
from crl_vehicle.models.heads import (
    LinearPresenceHead, LinearTypeHead, LinearProximityHead,
    MLPTypeHead, FullZTypeHead,
)
from crl_vehicle.losses.crl_loss import focal_cross_entropy
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

    VALID_PROBE_MODES = (
        "linear_ztype", "mlp_ztype", "linear_fullz",
        "linear_signal", "mlp_signal",
    )

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

        # Per-sensor Morlet derivation audit trail (populated by
        # _init_morlet_per_sensor; empty for other frontends). Persisted to
        # meta.json by the training entry points so runs are reproducible
        # without re-deriving from config.
        self._morlet_derived_params: dict[str, dict] = {}

        if config.frontend_bank in ("multiscale", "morlet", "morlet_learnable"):
            self._refresh_legacy_per_sensor_params(config)
            self.frontends, self.encoder, self.decoder, \
                self.encoders, self.decoders, self._morlet_derived_params = (
                    build_frontend(config, self.sensors)
                )
            head_keys = ["fused"] if config.frontend_fusion == "early" else self.sensors
        else:
            raise ValueError(f"Unknown frontend_bank: {config.frontend_bank!r}")

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

    def _refresh_legacy_per_sensor_params(self, config: CRLConfig) -> None:
        """Re-import legacy `morlet_per_sensor_params` into the unified
        `frontend_per_sensor_params` if the user mutated the legacy dict
        post-construction. Tests and ad-hoc scripts sometimes do
        `cfg.morlet_per_sensor_params[sensor] = ...` after CRLConfig() is
        built, which bypasses __post_init__ translation.

        Skipped for `frontend_type="morlet"` (the SR-heuristic variant) —
        that path uses synthesized params from the legacy heuristic, not
        the morlet_per_sensor_params dict. Refreshing would overwrite the
        synthesized values with the (irrelevant) default dict entries.
        """
        if config.frontend_bank not in ("morlet", "morlet_learnable"):
            return
        if config.frontend_type == "morlet":
            return
        legacy = config.morlet_per_sensor_params
        if not legacy:
            return
        config.frontend_per_sensor_params = {
            s: dict(p) for s, p in legacy.items()
        }
        # Early fusion: legacy morlet_fused always pooled to fused_seq_len,
        # overriding per-sensor target_tokens. Match that.
        if config.frontend_fusion == "early":
            for sp in config.frontend_per_sensor_params.values():
                sp["target_tokens"] = config.fused_seq_len
        else:
            # Late fusion: respect per-sensor target_tokens; default to 32.
            for sp in config.frontend_per_sensor_params.values():
                sp.setdefault("target_tokens", 32)

    def _build_type_head(self, d_z: int) -> nn.Module:
        if self.probe_mode == "linear_ztype":
            return LinearTypeHead()
        if self.probe_mode == "mlp_ztype":
            return MLPTypeHead()
        if self.probe_mode == "linear_fullz":
            return FullZTypeHead(d_z=d_z)
        if self.probe_mode == "linear_signal":
            # Reads z[0:d_signal] — the labeled subspace under the disentangled
            # 2-block partition. Sized by config.d_signal so the probe matches
            # the partition that produced the checkpoint.
            return LinearTypeHead(d_in=self.cfg.d_signal)
        if self.probe_mode == "mlp_signal":
            return MLPTypeHead(d_in=self.cfg.d_signal)
        raise ValueError(f"Unknown probe_mode: {self.probe_mode!r}")

    def is_fused_frontend(self) -> bool:
        """True when the frontend feeds a single shared encoder via
        time-concat (early fusion: multiscale, morlet_fused,
        morlet_learnable_fused) rather than per-sensor encoders."""
        return self.cfg.frontend_fusion == "early"

    def learnable_morlet_parameters(self) -> list[nn.Parameter]:
        """All nn.Parameters belonging to LearnableMorletFilterbank
        instances inside this model's frontends. Empty list for non-
        learnable variants. Used by Trainer to build a separate optimizer
        group with a lower LR multiplier."""
        params: list[nn.Parameter] = []
        for sensor_stack in self.frontends.values():
            for module in sensor_stack.modules():
                if isinstance(module, LearnableMorletFilterbank):
                    params.extend(module.parameters(recurse=False))
        return params


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
        """Backbone params — excludes downstream heads AND learnable Morlet
        params (those get their own optimizer group with a reduced LR)."""
        exclude_ids = set(
            id(p)
            for group in [
                self.pres_heads, self.type_heads, self.prox_heads,
                self.aux_pres_heads, self.aux_type_heads,
            ]
            for p in group.parameters()
        )
        exclude_ids.update(id(p) for p in self.learnable_morlet_parameters())
        return [p for p in self.parameters() if id(p) not in exclude_ids]

    def head_parameters(self) -> Iterator[nn.Parameter]:
        for group in [self.pres_heads, self.type_heads, self.prox_heads]:
            yield from group.parameters()

    def load_from_fixed_morlet_checkpoint(
        self, state_dict: dict, strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load weights from a fixed-Morlet (morlet_per_sensor or morlet_fused)
        checkpoint into this learnable model.

        Used for two-stage training: stage 1 converges a fixed-scale run,
        stage 2 loads that state here and fine-tunes scales as learnable.

        Conversion per sensor frontend:
          - kernel_re, kernel_im buffers exist on the source but not on the
            learnable model. Dropped from state_dict.
          - init_scales buffer is identical on both; loads directly.
          - log_scales parameter is initialized from log(init_scales). The
            source checkpoint has no log_scales key, so we inject one.
          - w0_per_filter parameter (only if learnable_w0=True) is
            initialized from the source's scalar w0 (stored on the bank
            instance, not in state_dict).
          - Downstream probe heads (pres_heads / type_heads / prox_heads)
            are dropped from source so they re-init fresh in stage 2.

        Returns (missing_keys, unexpected_keys) for logging. Raises if
        strict=True and any non-expected keys remain after conversion.
        """
        # Work on a copy — caller keeps their state_dict.
        source = dict(state_dict)

        # Drop downstream probe heads so they re-init fresh in stage 2.
        # aux_* heads DO carry over (they train during CRL).
        for key in list(source.keys()):
            if key.startswith(("pres_heads.", "type_heads.", "prox_heads.")):
                del source[key]

        # Per-sensor filterbank conversion: drop fixed kernels, inject
        # log_scales from init_scales.
        for sensor_name in self.frontends.keys():
            ks_re_key = f"frontends.{sensor_name}.0.kernel_re"
            ks_im_key = f"frontends.{sensor_name}.0.kernel_im"
            init_key  = f"frontends.{sensor_name}.0.init_scales"

            for k in (ks_re_key, ks_im_key):
                if k in source:
                    del source[k]

            log_scales_key = f"frontends.{sensor_name}.0.log_scales"
            if init_key in source and log_scales_key not in source:
                init_scales = source[init_key]
                source[log_scales_key] = torch.log(init_scales)

        missing, unexpected = self.load_state_dict(source, strict=False)

        # Classify the missing keys: log_scales / w0_per_filter entries for
        # sensors missing from the source are OK (fresh init). probe head
        # re-init is also expected (we dropped them).
        def _is_expected_missing(k: str) -> bool:
            return (
                k.startswith(("pres_heads.", "type_heads.", "prox_heads."))
                or k.endswith(".w0_per_filter")   # source had fixed w0; we init fresh
            )

        other_missing    = [k for k in missing    if not _is_expected_missing(k)]
        other_unexpected = [k for k in unexpected]

        if strict and (other_missing or other_unexpected):
            raise RuntimeError(
                f"Stage-2 checkpoint load mismatch. Missing: {other_missing}. "
                f"Unexpected: {other_unexpected}."
            )
        return missing, unexpected

    def _finetune_params(self, top_n: int) -> list[nn.Parameter]:
        """Return backbone parameters to unfreeze for fine-tuning.

        top_n == -1: entire backbone.
        top_n >= 1:  top N transformer layers + mu/lv projection heads.
        """
        if top_n == -1:
            return list(self.backbone_parameters())

        params: list[nn.Parameter] = []
        if self.is_fused_frontend():
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
        stage2: bool = False,
    ) -> None:
        self.model  = model
        self.cfg    = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stage2 = stage2

        self.mode: TrainingMode = build_training_mode(config).to(device)
        self.ckpt_state = CheckpointState()

        self.beta = 0.0
        self._pres_pos_weight:    torch.Tensor | None = None
        self._type_class_weights: torch.Tensor | None = None

        # The optimizer covers: (1) model backbone, (2) learnable params
        # carried by the mode (e.g. ConditionalPrior MLP at Checkpoint 2),
        # (3) learnable Morlet filterbank params — separate group at reduced
        # LR so the init-near-optimal filterbank doesn't wander.
        #
        # Stage 2 (user loaded a converged fixed-Morlet checkpoint via
        # --init-from-run): the backbone was already trained, so we drop its
        # LR to stage2_encoder_lr_mult× the base LR. Filters become the
        # primary mover; encoder just fine-tunes.
        backbone_param_ids = {id(p) for p in model.backbone_parameters()}
        mode_params = [p for p in self.mode.parameters()
                       if id(p) not in backbone_param_ids]
        learnable_morlet_params = model.learnable_morlet_parameters()

        backbone_lr = config.lr
        if stage2:
            backbone_lr = config.lr * config.stage2_encoder_lr_mult

        param_groups = [
            {"params": model.backbone_parameters(), "weight_decay": config.wd,
             "lr": backbone_lr, "name": "backbone"}
        ]
        if mode_params:
            param_groups.append({
                "params": mode_params, "weight_decay": config.wd,
                "lr": backbone_lr, "name": "mode",
            })
        if learnable_morlet_params:
            param_groups.append({
                "params":       learnable_morlet_params,
                "weight_decay": config.wd,
                "lr":           config.lr * config.morlet_learnable_lr_mult,
                "name":         "learnable_morlet",
            })
        self.optimizer = torch.optim.AdamW(param_groups, lr=config.lr)

        if stage2:
            self.scheduler = self._build_stage2_scheduler(
                total_epochs=config.n_epochs, warmup_epochs=3,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr_min
            )

    def _build_stage2_scheduler(
        self, total_epochs: int, warmup_epochs: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Stage-2 schedule: for the learnable_morlet group, linear warmup
        from 0 to full LR over warmup_epochs, then cosine annealing over the
        remaining epochs. Backbone and mode groups start at their already-
        reduced stage-2 LR and do plain cosine annealing over total_epochs.

        Warmup protects the converged encoder from an immediate perturbation
        when the filters start moving from their fixed-scale init.
        """
        cfg = self.cfg
        lr_min_ratio = max(cfg.lr_min / max(cfg.lr, 1e-12), 0.0)

        def _lr_multiplier_cosine(epoch: int, t_max: int) -> float:
            if t_max <= 0:
                return 1.0
            progress = min(max(epoch / t_max, 0.0), 1.0)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor

        def _lambda_for_group(group_name: str):
            def _fn(epoch: int) -> float:
                if group_name == "learnable_morlet" and epoch < warmup_epochs:
                    return (epoch + 1) / max(warmup_epochs, 1)
                # Cosine over the post-warmup window for filters, over full
                # schedule for backbone/mode. Subtracting warmup_epochs on
                # the filter group keeps its peak at 1.0 right after warmup.
                if group_name == "learnable_morlet":
                    return _lr_multiplier_cosine(
                        epoch - warmup_epochs,
                        max(total_epochs - warmup_epochs, 1),
                    )
                return _lr_multiplier_cosine(epoch, total_epochs)
            return _fn

        lambdas = [
            _lambda_for_group(g.get("name", ""))
            for g in self.optimizer.param_groups
        ]
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambdas)

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

    def _log_learnable_morlet_freqs(self, epoch: int) -> None:
        """Append one row per (sensor, filter_idx) to learnable_morlet_freqs.csv
        with current learned frequencies. No-op for non-learnable variants.
        Writes header on first call."""
        banks = {
            sensor: module
            for sensor, stack in self.model.frontends.items()
            for module in stack.modules()
            if isinstance(module, LearnableMorletFilterbank)
        }
        if not banks:
            return
        csv_path = self.save_dir / "learnable_morlet_freqs.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "sensor", "filter_idx", "freq_hz", "w0"])
            for sensor, bank in banks.items():
                freqs = bank.current_frequencies().cpu().tolist()
                if bank.learnable_w0:
                    w0_vec = bank.w0_per_filter.detach().cpu().tolist()
                else:
                    w0_vec = [bank.w0] * bank.out_channels
                for idx, (f_hz, w) in enumerate(zip(freqs, w0_vec)):
                    writer.writerow([epoch, sensor, idx, f_hz, w])

    def _learned_morlet_params_summary(self) -> dict:
        """Snapshot of final learned Morlet params for meta.json. Returns
        {} for non-learnable variants — entry points write this alongside
        the derived-params block."""
        out: dict = {}
        for sensor, stack in self.model.frontends.items():
            for module in stack.modules():
                if isinstance(module, LearnableMorletFilterbank):
                    block = {"freq_hz": module.current_frequencies().cpu().tolist()}
                    if module.learnable_w0:
                        block["w0"] = module.w0_per_filter.detach().cpu().tolist()
                    out[sensor] = block
                    break
        return out

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

                self._log_learnable_morlet_freqs(epoch)

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
        summary = self.mode.checkpoint_summary(self.ckpt_state)
        learned = self._learned_morlet_params_summary()
        if learned:
            summary["learned_morlet_params"] = learned
        summary_path = self.save_dir / "crl_checkpoint_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

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
        cfg = model.cfg

        use_fullz  = model.probe_mode == "linear_fullz"
        use_signal = model.probe_mode in ("linear_signal", "mlp_signal")
        d_signal   = self.model.cfg.d_signal

        def _select_type_slice(z_full, z_type_block, mask):
            if use_fullz:
                return z_full[mask]
            if use_signal:
                return z_full[mask][..., :d_signal]
            return z_type_block[mask]

        def _type_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if cfg.use_focal_type:
                return focal_cross_entropy(
                    logits, target, weight=tcw, gamma=cfg.focal_type_gamma,
                )
            return torch.nn.functional.cross_entropy(logits, target, weight=tcw)

        if model.is_fused_frontend():
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
                    z_for_type = _select_type_slice(z, z_type, valid)
                    type_logit = model.type_heads["fused"](z_for_type)
                    total = total + _type_loss(type_logit, vtype[valid])
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
                    z_for_type = _select_type_slice(z, z_type, valid)
                    type_logit = model.type_heads[sensor](z_for_type)
                    total = total + _type_loss(type_logit, vtype[valid])
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

