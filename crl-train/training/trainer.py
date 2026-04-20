"""CRLModel assembly and Trainer for CRL pre-training and downstream evaluation."""
from __future__ import annotations

import csv
import copy
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.models.frontend import MultiScale1DFrontend, MorletFilterbank
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder
from crl_vehicle.models.intervention import (
    UnknownInterventionClassifier, label_change_target
)
from crl_vehicle.models.heads import (
    LinearPresenceHead, LinearTypeHead, LinearProximityHead
)
from crl_vehicle.losses.crl_loss import (
    reconstruction_loss, kl_divergence, intervention_matching_loss
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

    def __init__(
        self,
        config: CRLConfig,
        sensors: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.sensors = sensors or ["audio", "seismic"]
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
            self.type_heads[key]     = LinearTypeHead()
            self.prox_heads[key]     = LinearProximityHead()
            self.aux_pres_heads[key] = nn.Linear(CausalLatentSpace.D_PRES, 1)
            self.aux_type_heads[key] = nn.Linear(CausalLatentSpace.D_TYPE, 4)

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
        for sensor in self.sensors:
            mc = config.modality_cfg(sensor)
            ks = config.morlet_kernel_size
            self.frontends[sensor] = nn.Sequential(
                MorletFilterbank(mc.n_channels, config.d_model, ks, mc.sample_rate),
                nn.AvgPool1d(stride, stride),
            )
            seq_len = mc.window_size // stride
            self.encoders[sensor] = TemporalEncoder(
                in_channels=config.d_model,
                d_z=d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            self.decoders[sensor] = FeatureDecoder(
                out_channels=config.d_model,
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Handles CRL pre-training and downstream head training."""

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

        self.beta = 0.0
        self.prev_val_recon = float("inf")

        self.optimizer = torch.optim.AdamW(
            model.backbone_parameters(),
            lr=config.lr,
            weight_decay=config.wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.n_epochs, eta_min=config.lr_min
        )

    # ------------------------------------------------------------------
    # Beta annealing
    # ------------------------------------------------------------------

    def _update_beta(self, val_m: dict) -> str:
        raw_kl = val_m["val_raw_kl"]
        recon_improving = (
            val_m["val_recon"] < self.prev_val_recon - self.cfg.recon_min_delta
        )
        self.prev_val_recon = val_m["val_recon"]
        if raw_kl < self.cfg.kl_floor:
            self.beta = max(0.0, self.beta - self.cfg.beta_step)
            return "↓collapse"
        elif recon_improving or raw_kl > self.cfg.kl_target:
            self.beta = min(1.0, self.beta + self.cfg.beta_step)
            return "↑"
        return "→hold"

    # ------------------------------------------------------------------
    # Forward pair
    # ------------------------------------------------------------------

    def _forward_pair(
        self, batch: dict, beta: float
    ) -> tuple[torch.Tensor, dict]:
        if self.cfg.frontend_type == "multiscale":
            return self._forward_pair_fused(batch, beta)
        return self._forward_pair_per_sensor(batch, beta)

    def _forward_pair_fused(
        self, batch: dict, beta: float
    ) -> tuple[torch.Tensor, dict]:
        model = self.model
        cfg   = self.cfg
        dev   = self.device

        avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
        if not avail.any():
            zero = torch.tensor(0.0, device=dev, requires_grad=True)
            return zero, {"recon": 0.0, "kl": 0.0, "raw_kl": 0.0, "interv": 0.0, "total": 0.0}

        x_a = batch["x_audio_t"][avail].to(dev)
        x_s = batch["x_seismic_t"][avail].to(dev)

        features, z_t, mu_t, lv_t = model.encode_fused(x_a, x_s)
        x_hat = model.decode_fused(z_t)

        recon   = reconstruction_loss(x_hat, features.detach())
        kl      = kl_divergence(mu_t, lv_t, beta=beta)
        raw_kl  = kl_divergence(mu_t, lv_t, beta=1.0)

        # Aux supervised losses on anchor
        z_pres, z_type, z_prox, z_env, _ = model.latent.split(z_t)
        det_t  = batch["detection_label_t"][avail].float().to(dev)
        type_t = batch["vehicle_type_t"][avail].to(dev)

        aux_pres_logit = model.aux_pres_heads["fused"](z_pres).squeeze(-1)
        aux_pres = torch.nn.functional.binary_cross_entropy_with_logits(aux_pres_logit, det_t)

        valid_type = type_t >= 0
        aux_type = torch.tensor(0.0, device=dev)
        if valid_type.any():
            aux_type = torch.nn.functional.cross_entropy(
                model.aux_type_heads["fused"](z_type[valid_type]),
                type_t[valid_type],
            )

        # Intervention: use consecutive partner (p0)
        n_partners = batch["n_partners"]
        interv = torch.tensor(0.0, device=dev)
        if n_partners > 0:
            x_a_p0 = batch["x_audio_p0"][avail].to(dev)
            x_s_p0 = batch["x_seismic_p0"][avail].to(dev)
            _, z_tn, _, _ = model.encode_fused(x_a_p0, x_s_p0)
            _, _, _, z_env_tn, _ = model.latent.split(z_tn)
            det_tn   = batch["detection_label_p0"][avail].to(dev)
            type_tn  = batch["vehicle_type_p0"][avail].to(dev)
            targets  = label_change_target(det_t.long(), det_tn, type_t, type_tn).to(dev)
            logits   = model.interv_classifier(z_env, z_env_tn)
            interv   = intervention_matching_loss(logits, targets)

        total = (recon + kl
                 + cfg.lambda_interv   * interv
                 + cfg.lambda_aux_pres * aux_pres
                 + cfg.lambda_aux_type * aux_type)

        metrics = {
            "recon":   recon.item(),
            "kl":      kl.item(),
            "raw_kl":  raw_kl.item(),
            "interv":  interv.item() if isinstance(interv, torch.Tensor) else interv,
            "total":   total.item(),
        }
        return total, metrics

    def _forward_pair_per_sensor(
        self, batch: dict, beta: float
    ) -> tuple[torch.Tensor, dict]:
        model = self.model
        cfg   = self.cfg
        dev   = self.device

        total_loss = torch.tensor(0.0, device=dev)
        agg: dict[str, float] = {"recon": 0.0, "kl": 0.0, "raw_kl": 0.0,
                                  "interv": 0.0, "total": 0.0}
        n_active = 0

        for sensor in model.sensors:
            avail_key = f"{sensor}_avail"
            avail = batch[avail_key].bool()
            if not avail.any():
                continue

            x_key = f"x_{sensor}_t"
            x = batch[x_key][avail].to(dev)
            features, z_t, mu_t, lv_t = model.encode(sensor, x)
            x_hat = model.decode(sensor, z_t)

            recon  = reconstruction_loss(x_hat, features.detach())
            kl     = kl_divergence(mu_t, lv_t, beta=beta)
            raw_kl = kl_divergence(mu_t, lv_t, beta=1.0)

            z_pres, z_type, z_prox, z_env, _ = model.latent.split(z_t)
            det_t  = batch["detection_label_t"][avail].float().to(dev)
            type_t = batch["vehicle_type_t"][avail].to(dev)

            aux_pres_logit = model.aux_pres_heads[sensor](z_pres).squeeze(-1)
            aux_pres = torch.nn.functional.binary_cross_entropy_with_logits(
                aux_pres_logit, det_t
            )

            valid_type = type_t >= 0
            aux_type = torch.tensor(0.0, device=dev)
            if valid_type.any():
                aux_type = torch.nn.functional.cross_entropy(
                    model.aux_type_heads[sensor](z_type[valid_type]),
                    type_t[valid_type],
                )

            interv = torch.tensor(0.0, device=dev)
            n_partners = batch["n_partners"]
            if n_partners > 0:
                x_p0 = batch[f"x_{sensor}_p0"][avail].to(dev)
                _, z_tn, _, _ = model.encode(sensor, x_p0)
                _, _, _, z_env_tn, _ = model.latent.split(z_tn)
                det_tn  = batch["detection_label_p0"][avail].to(dev)
                type_tn = batch["vehicle_type_p0"][avail].to(dev)
                targets = label_change_target(det_t.long(), det_tn, type_t, type_tn).to(dev)
                logits  = model.interv_classifier(z_env, z_env_tn)
                interv  = intervention_matching_loss(logits, targets)

            sensor_loss = (recon + kl
                           + cfg.lambda_interv   * interv
                           + cfg.lambda_aux_pres * aux_pres
                           + cfg.lambda_aux_type * aux_type)
            total_loss = total_loss + sensor_loss
            agg["recon"]  += recon.item()
            agg["kl"]     += kl.item()
            agg["raw_kl"] += raw_kl.item()
            agg["interv"] += interv.item() if isinstance(interv, torch.Tensor) else interv
            agg["total"]  += sensor_loss.item()
            n_active += 1

        if n_active > 1:
            total_loss = total_loss / n_active
            for k in agg:
                agg[k] /= n_active

        if n_active == 0:
            total_loss = torch.tensor(0.0, device=dev, requires_grad=True)

        return total_loss, agg

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, beta: float, steps_per_epoch: int | None = None
    ) -> dict:
        self.model.train()
        agg: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(loader):
            if steps_per_epoch is not None and i >= steps_per_epoch:
                break
            self.optimizer.zero_grad()
            loss, metrics = self._forward_pair(batch, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {f"train_{k}": v / max(n, 1) for k, v in agg.items()}

    def _eval_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for batch in loader:
                _, metrics = self._forward_pair(batch, beta=1.0)
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
        out = {f"val_{k}": v / max(n, 1) for k, v in agg.items()}
        # ref_elbo = recon + KL (beta=1) — epoch-invariant checkpoint metric
        out["val_ref_elbo"] = out.get("val_recon", 0.0) + out.get("val_raw_kl", 0.0)
        return out

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
        best_ref_elbo = float("inf")
        best_state: dict | None = None
        patience_count = 0
        metrics_rows: list[dict] = []

        for epoch in range(epochs):
            train_m = self._train_epoch(train_loader, self.beta, steps_per_epoch)
            val_m   = self._eval_epoch(val_loader, self.beta)
            self.scheduler.step()

            event = self._update_beta(val_m)
            ref_elbo = val_m["val_ref_elbo"]

            row = {"epoch": epoch, "beta": self.beta, "beta_event": event}
            row.update(train_m)
            row.update(val_m)
            metrics_rows.append(row)

            print(
                f"Epoch {epoch:3d} | beta={self.beta:.3f} {event} | "
                f"recon={val_m.get('val_recon',0):.4f} kl={val_m.get('val_raw_kl',0):.4f} "
                f"ref_elbo={ref_elbo:.4f}"
            )

            if ref_elbo < best_ref_elbo - 1e-5:
                best_ref_elbo = ref_elbo
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.save_dir / "crl_best.pth")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")
        self._save_csv(metrics_rows, self.save_dir / "crl_metrics.csv")

    # ------------------------------------------------------------------
    # Downstream training
    # ------------------------------------------------------------------

    def train_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> None:
        # Load best CRL backbone if available
        best_path = self.save_dir / "crl_best.pth"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.head_parameters():
            p.requires_grad_(True)

        head_opt = torch.optim.AdamW(self.model.head_parameters(), lr=self.cfg.lr)

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                head_opt.zero_grad()
                loss = self._downstream_loss(batch)
                loss.backward()
                head_opt.step()

        # Unfreeze
        for p in self.model.parameters():
            p.requires_grad_(True)

    def _downstream_loss(self, batch: dict) -> torch.Tensor:
        model = self.model
        dev   = self.device
        total = torch.tensor(0.0, device=dev)
        n = 0

        if self.cfg.frontend_type == "multiscale":
            avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
            if avail.any():
                x_a = batch["x_audio"][avail].to(dev)
                x_s = batch["x_seismic"][avail].to(dev)
                _, z, _, _ = model.encode_fused(x_a, x_s)
                z_pres, z_type, _, _, _ = model.latent.split(z)
                det   = batch["detection_label"][avail].float().to(dev)
                vtype = batch["vehicle_type"][avail].to(dev)
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                    model.pres_heads["fused"](z_pres).squeeze(-1), det
                )
                valid = vtype >= 0
                if valid.any():
                    total = total + torch.nn.functional.cross_entropy(
                        model.type_heads["fused"](z_type[valid]), vtype[valid]
                    )
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
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                    model.pres_heads[sensor](z_pres).squeeze(-1), det
                )
                valid = vtype >= 0
                if valid.any():
                    total = total + torch.nn.functional.cross_entropy(
                        model.type_heads[sensor](z_type[valid]), vtype[valid]
                    )
                n += 1

        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _save_csv(rows: list[dict], path: Path) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
