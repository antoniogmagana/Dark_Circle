"""
Trainer for the CITRIS-style CRL pipeline.

CRL pre-training (ConsecutivePairDataset):
    Per-modality per time step:
        frontend → TemporalEncoder → CausalLatentSpace.split()
                                   → FeatureDecoder

    Loss = reconstruction_loss + beta * kl_divergence
         + lambda_interv * intervention_matching_loss

    Checkpointing uses fixed-reference ELBO (beta=1) so the metric is
    epoch-invariant even when beta is annealed during training.

Downstream training (SensorDataset, frozen backbone):
    LinearPresenceHead on z_pres
    LinearTypeHead on z_type
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES, CLASS_MAP
from crl_vehicle.models.frontend import MultiScale1DFrontend, LearnableMorlet1D
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.intervention import UnknownInterventionClassifier, interv_idx_to_block_target
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead
from crl_vehicle.losses.crl_loss import reconstruction_loss, kl_divergence, intervention_matching_loss


# ---------------------------------------------------------------------------
# CRLModel
# ---------------------------------------------------------------------------

class CRLModel(nn.Module):
    """
    Full CITRIS CRL model. One pipeline per modality:
        MultiScale1DFrontend → TemporalEncoder → CausalLatentSpace → FeatureDecoder

    Shared across modalities:
        UnknownInterventionClassifier

    Downstream heads (separately trained, frozen backbone):
        LinearPresenceHead, LinearTypeHead per modality
    """

    def __init__(self, config: CRLConfig, sensors: list | None = None):
        super().__init__()
        self.cfg = config
        self.sensors = sensors or MODALITIES

        self.frontends   = nn.ModuleDict()
        self.encoders    = nn.ModuleDict()
        self.decoders    = nn.ModuleDict()
        self.pres_heads  = nn.ModuleDict()
        self.type_heads  = nn.ModuleDict()

        self.latent = CausalLatentSpace(d_z=10)
        self.interv_classifier = UnknownInterventionClassifier(d_z=10)

        for sensor in self.sensors:
            mod_cfg = config.modality_cfg(sensor)

            if self.cfg.frontend_type == "multiscale":
                frontend = MultiScale1DFrontend(
                    in_channels=mod_cfg.n_channels,
                    out_channels=config.d_model,
                )
            elif self.cfg.frontend_type == "morlet":
                frontend = LearnableMorlet1D(
                    in_channels=mod_cfg.n_channels,
                    out_channels=config.d_model,
                    kernel_size=config.morlet_kernel_size,
                )
            else:
                raise ValueError(f"Unknown frontend_type: {self.cfg.frontend_type}")
            self.frontends[sensor] = frontend

            # Infer frontend output shape with a dummy pass
            with torch.no_grad():
                dummy = torch.zeros(1, mod_cfg.n_channels, mod_cfg.window_size)
                feat_shape = frontend(dummy).shape   # (1, C_out, T')
            c_out, t_prime = feat_shape[1], feat_shape[2]

            self.encoders[sensor] = TemporalEncoder(
                in_channels=c_out,
                d_z=10,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            self.decoders[sensor] = FeatureDecoder(
                out_channels=c_out,
                seq_len=t_prime,
                d_z=10,
                d_model=config.d_model,
            )

            self.pres_heads[sensor] = LinearPresenceHead(d_in=1)
            self.type_heads[sensor] = LinearTypeHead(d_in=4, n_classes=len(CLASS_MAP))

    def encode(self, sensor: str, x: torch.Tensor):
        """
        Run frontend → encoder for one modality.
        Returns (features, z, mu, logvar).
        features is kept so the caller can compute reconstruction loss.
        """
        features = self.frontends[sensor](x)          # (B, C, T')
        z, mu, logvar = self.encoders[sensor](features)
        return features, z, mu, logvar

    def decode(self, sensor: str, z: torch.Tensor) -> torch.Tensor:
        return self.decoders[sensor](z)               # (B, C, T')

    def backbone_parameters(self):
        """All parameters except downstream heads."""
        head_params = set(
            list(self.pres_heads.parameters()) +
            list(self.type_heads.parameters())
        )
        return [p for p in self.parameters() if p not in head_params]

    def head_parameters(self):
        return list(self.pres_heads.parameters()) + list(self.type_heads.parameters())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(
        self,
        model: CRLModel,
        config: CRLConfig,
        device: torch.device,
        save_dir: Path,
    ):
        self.model    = model
        self.cfg      = config
        self.device   = device
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.backbone_parameters(),
            lr=config.lr,
            weight_decay=config.wd,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs,
            eta_min=config.lr_min,
        )
        self.best_ref_elbo = float("inf")
        self.patience_ctr  = 0
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # CRL pre-training
    # ------------------------------------------------------------------

    def train_crl(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        fieldnames = [
            "epoch", "beta",
            "train_recon", "train_kl", "train_interv", "train_total",
            "val_recon",   "val_kl",   "val_interv",   "val_total",
            "val_ref_elbo",   # fixed-reference ELBO (beta=1), used for checkpointing
        ]
        metrics_path = self.save_dir / "crl_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            # Linear beta warmup from 0 → 1 over first half of training
            beta = min(1.0, 2.0 * epoch / max(epochs, 1))

            train_m = self._train_epoch(train_loader, beta)
            val_m   = self._eval_epoch(val_loader, beta)
            self.scheduler.step()

            row = {"epoch": epoch, "beta": round(beta, 4), **train_m, **val_m}
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | beta={beta:.2f} | "
                f"tr={train_m['train_total']:.4f} "
                f"val={val_m['val_total']:.4f} "
                f"ref_elbo={val_m['val_ref_elbo']:.4f}"
            )

            ref_elbo = val_m["val_ref_elbo"]
            if ref_elbo < self.best_ref_elbo:
                self.best_ref_elbo = ref_elbo
                self.patience_ctr  = 0
                torch.save(self.model.state_dict(), self.save_dir / "crl_best.pth")
                print(f"  New best (ref_elbo={ref_elbo:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")

    def _forward_pair(self, batch: dict, beta: float) -> tuple[torch.Tensor, dict]:
        """Shared forward pass for train and eval. Returns (loss, metrics_dict)."""
        recon_total = kl_total = interv_total = n_mod = 0.0

        for sensor in self.model.sensors:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue

            x_t  = batch[f"x_{sensor}_t"][avail].to(self.device)
            x_tn = batch[f"x_{sensor}_tn"][avail].to(self.device)

            feat_t, z_t, mu_t, lv_t = self.model.encode(sensor, x_t)
            assert mu_t.isfinite().all(), f"mu_t became non-finite for sensor {sensor}"
            assert lv_t.isfinite().all(), f"lv_t became non-finite for sensor {sensor}"

            feat_tn, z_tn, mu_tn, lv_tn = self.model.encode(sensor, x_tn)
            assert mu_tn.isfinite().all(), f"mu_tn became non-finite for sensor {sensor}"
            assert lv_tn.isfinite().all(), f"lv_tn became non-finite for sensor {sensor}"

            x_hat_t  = self.model.decode(sensor, z_t)
            x_hat_tn = self.model.decode(sensor, z_tn)

            recon = (reconstruction_loss(x_hat_t, feat_t) +
                     reconstruction_loss(x_hat_tn, feat_tn)) / 2

            kl_t = kl_divergence(mu_t, lv_t, beta=beta)
            assert kl_t.isfinite(), f"kl_t became non-finite for sensor {sensor}"
            kl_tn = kl_divergence(mu_tn, lv_tn, beta=beta)
            assert kl_tn.isfinite(), f"kl_tn became non-finite for sensor {sensor}"
            kl = (kl_t + kl_tn) / 2

            # Intervention matching
            interv_idx_t  = batch["interv_idx_t"][avail].to(self.device)
            interv_idx_tn = batch["interv_idx_tn"][avail].to(self.device)
            targets = interv_idx_to_block_target(interv_idx_t, interv_idx_tn)
            logits  = self.model.interv_classifier(z_t, z_tn)
            interv  = intervention_matching_loss(logits, targets)

            recon_total  += recon
            kl_total     += kl
            interv_total += interv
            n_mod        += 1

        if n_mod == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {"recon": 0.0, "kl": 0.0, "interv": 0.0, "total": 0.0}

        recon_total  /= n_mod
        kl_total     /= n_mod
        interv_total /= n_mod
        total = recon_total + kl_total + self.cfg.lambda_interv * interv_total

        return total, {
            "recon":  recon_total.item(),
            "kl":     kl_total.item(),
            "interv": interv_total.item(),
            "total":  total.item(),
        }

    def _train_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.train()
        accum: dict[str, float] = {}
        n = 0
        for batch in loader:
            # Sanity check for NaNs in the input data
            for sensor in self.model.sensors:
                if batch[f"{sensor}_avail"].any():
                    assert batch[f"x_{sensor}_t"].isfinite().all(), f"NaNs detected in input x_{sensor}_t"
                    assert batch[f"x_{sensor}_tn"].isfinite().all(), f"NaNs detected in input x_{sensor}_tn"

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss, metrics = self._forward_pair(batch, beta)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            if self.cfg.steps_per_epoch and n >= self.cfg.steps_per_epoch:
                break
        return {f"train_{k}": v / max(n, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.eval()
        accum: dict[str, float] = {}
        ref_accum: dict[str, float] = {}
        n = 0
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                _, metrics      = self._forward_pair(batch, beta)
                _, ref_metrics  = self._forward_pair(batch, beta=1.0)
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            ref_accum["total"] = ref_accum.get("total", 0.0) + ref_metrics["total"]
            n += 1
        self.model.train()
        out = {f"val_{k}": v / max(n, 1) for k, v in accum.items()}
        out["val_ref_elbo"] = ref_accum.get("total", 0.0) / max(n, 1)
        return out

    # ------------------------------------------------------------------
    # Downstream head training
    # ------------------------------------------------------------------

    def train_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        # Freeze backbone
        for p in self.model.backbone_parameters():
            p.requires_grad = False

        for sensor in self.model.sensors:
            self._train_sensor_heads(sensor, train_loader, val_loader, epochs)

        # Unfreeze for any subsequent use
        for p in self.model.backbone_parameters():
            p.requires_grad = True

    def _train_sensor_heads(
        self,
        sensor: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        pres_head = self.model.pres_heads[sensor]
        type_head = self.model.type_heads[sensor]

        opt = torch.optim.AdamW(
            list(pres_head.parameters()) + list(type_head.parameters()),
            lr=self.cfg.lr,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

        metrics_path = self.save_dir / f"downstream_metrics_{sensor}.csv"
        fieldnames = ["epoch", "train_pres_loss", "train_type_loss",
                      "val_pres_acc", "val_pres_f1", "val_type_acc", "val_type_f1"]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        best_pres_f1 = best_type_f1 = 0.0
        patience_pres = patience_type = 0
        stopped_pres = stopped_type = False

        print(f"\n=== Downstream [{sensor}] ===")
        for epoch in range(epochs):
            if stopped_pres and stopped_type:
                break

            self.model.eval()
            pres_head.train()
            type_head.train()

            ep_pres = ep_type = 0.0
            n_pres = n_type = 0

            for batch in train_loader:
                avail = batch[f"{sensor}_avail"]
                if not avail.any():
                    continue
                x = batch[f"x_{sensor}"][avail].to(self.device)
                det   = batch["detection_label"][avail].to(self.device)
                vtype = batch["vehicle_type"][avail].to(self.device)

                with torch.no_grad():
                    _, z, _, _ = self.model.encode(sensor, x)
                z_pres, z_type, _, _ = self.model.latent.split(z)

                opt.zero_grad()
                loss = torch.tensor(0.0, device=self.device)

                if not stopped_pres:
                    pres_logit = pres_head(z_pres).squeeze(-1)
                    pres_loss  = F.binary_cross_entropy_with_logits(
                        pres_logit, det.float()
                    )
                    loss = loss + pres_loss
                    ep_pres += pres_loss.item()
                    n_pres  += 1

                type_mask = (det == 1) & (vtype >= 0)
                if not stopped_type and type_mask.any():
                    type_logits = type_head(z_type[type_mask])
                    type_loss   = F.cross_entropy(type_logits, vtype[type_mask])
                    loss = loss + type_loss
                    ep_type += type_loss.item()
                    n_type  += 1

                if loss.requires_grad:
                    loss.backward()
                    opt.step()

            val_m = self._eval_downstream(val_loader, sensor)
            sched.step(-(val_m["val_pres_f1"] + val_m["val_type_f1"]))

            row = {
                "epoch":           epoch,
                "train_pres_loss": ep_pres / max(n_pres, 1),
                "train_type_loss": ep_type / max(n_type, 1),
                **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            status = []
            if val_m["val_pres_f1"] > best_pres_f1:
                best_pres_f1 = val_m["val_pres_f1"]
                patience_pres = 0
                torch.save(pres_head.state_dict(), self.save_dir / f"pres_head_{sensor}_best.pth")
                status.append(f"pres✓ F1={best_pres_f1:.4f}")
            elif not stopped_pres:
                patience_pres += 1
                if patience_pres >= self.cfg.early_stop_patience:
                    stopped_pres = True
                    status.append(f"pres stopped (ep {epoch})")

            if val_m["val_type_f1"] > best_type_f1:
                best_type_f1 = val_m["val_type_f1"]
                patience_type = 0
                torch.save(type_head.state_dict(), self.save_dir / f"type_head_{sensor}_best.pth")
                status.append(f"type✓ F1={best_type_f1:.4f}")
            elif not stopped_type:
                patience_type += 1
                if patience_type >= self.cfg.early_stop_patience:
                    stopped_type = True
                    status.append(f"type stopped (ep {epoch})")

            print(
                f"Epoch {epoch:3d} | "
                f"pres={row['train_pres_loss']:.4f} type={row['train_type_loss']:.4f} | "
                f"pres_f1={val_m['val_pres_f1']:.4f} type_f1={val_m['val_type_f1']:.4f}"
                + (f"  [{', '.join(status)}]" if status else "")
            )

        print(f"[{sensor}] Done. Best pres F1={best_pres_f1:.4f}  type F1={best_type_f1:.4f}")

    @torch.no_grad()
    def _eval_downstream(self, loader: DataLoader, sensor: str) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()
        pres_head = self.model.pres_heads[sensor]
        type_head = self.model.type_heads[sensor]

        det_true, det_pred = [], []
        cls_true, cls_pred = [], []

        for batch in loader:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue
            x = batch[f"x_{sensor}"][avail].to(self.device)
            det   = batch["detection_label"][avail]
            vtype = batch["vehicle_type"][avail]

            _, z, _, _ = self.model.encode(sensor, x)
            z_pres, z_type, _, _ = self.model.latent.split(z)

            pres_logit  = pres_head(z_pres).squeeze(-1)
            type_logits = type_head(z_type)

            det_pred.extend((pres_logit > 0).cpu().long().tolist())
            det_true.extend(det.tolist())

            type_mask = (det == 1) & (vtype >= 0)
            if type_mask.any():
                cls_pred.extend(type_logits[type_mask.to(self.device)].argmax(1).cpu().tolist())
                cls_true.extend(vtype[type_mask].tolist())

        def _f1(true, pred):
            if not true:
                return 0.0, 0.0
            return (
                accuracy_score(true, pred),
                f1_score(true, pred, average="weighted", zero_division=0),
            )

        pres_acc, pres_f1 = _f1(det_true, det_pred)
        type_acc, type_f1 = _f1(cls_true, cls_pred)
        self.model.train()
        return {
            "val_pres_acc": pres_acc,
            "val_pres_f1":  pres_f1,
            "val_type_acc": type_acc,
            "val_type_f1":  type_f1,
        }
