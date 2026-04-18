"""
Trainer for the CITRIS-style CRL pipeline.

CRL pre-training (ConsecutivePairDataset):
    Per-modality per time step:
        frontend → TemporalEncoder → CausalLatentSpace.split()
                                   → FeatureDecoder

    Loss = reconstruction_loss + beta * kl_divergence
         + lambda_interv * intervention_matching_loss
         + lambda_aux_* * aux_supervision_losses

    Checkpointing uses fixed-reference ELBO (beta=1) so the metric is
    epoch-invariant even when beta is annealed during training.

Downstream training (SensorDataset, frozen backbone):
    LinearPresenceHead on z_pres
    LinearTypeHead on z_type
    LinearProximityHead on z_prox
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES, CLASS_MAP
from crl_vehicle.models.frontend import MultiScale1DFrontend, LearnableMorlet1D
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.intervention import UnknownInterventionClassifier, label_change_target
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead, LinearProximityHead
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
        LinearPresenceHead, LinearTypeHead, LinearProximityHead per modality

    Auxiliary heads (used only in pre-training for subspace supervision):
        aux_pres_heads, aux_type_heads, aux_prox_heads
    """

    def __init__(self, config: CRLConfig, sensors: list | None = None):
        super().__init__()
        self.cfg = config
        self.sensors = sensors or MODALITIES
        d_z = config.d_z   # 24

        self.frontends   = nn.ModuleDict()
        self.encoders    = nn.ModuleDict()
        self.decoders    = nn.ModuleDict()
        self.pres_heads  = nn.ModuleDict()
        self.type_heads  = nn.ModuleDict()
        self.prox_heads  = nn.ModuleDict()

        # Auxiliary heads for pre-training supervision (discarded after CRL phase)
        self.aux_pres_heads = nn.ModuleDict()
        self.aux_type_heads = nn.ModuleDict()

        self.latent = CausalLatentSpace(d_z=d_z)
        self.interv_classifier = UnknownInterventionClassifier(
            d_env=CausalLatentSpace.D_ENV
        )

        for sensor in self.sensors:
            mod_cfg = config.modality_cfg(sensor)

            if self.cfg.frontend_type == "multiscale":
                pool_stride = config.multiscale_pool_stride
                frontend = nn.Sequential(
                    MultiScale1DFrontend(
                        in_channels=mod_cfg.n_channels,
                        out_channels=config.d_model,
                    ),
                    nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride),
                )
            elif self.cfg.frontend_type == "morlet":
                pool_stride = config.morlet_pool_stride
                frontend = nn.Sequential(
                    LearnableMorlet1D(
                        in_channels=mod_cfg.n_channels,
                        out_channels=config.d_model,
                        kernel_size=config.morlet_kernel_size,
                        sample_rate=mod_cfg.sample_rate,
                    ),
                    nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride),
                )
            else:
                raise ValueError(f"Unknown frontend_type: {self.cfg.frontend_type}")
            self.frontends[sensor] = frontend

            with torch.no_grad():
                dummy = torch.zeros(1, mod_cfg.n_channels, mod_cfg.window_size)
                feat_shape = frontend(dummy).shape
            c_out, t_prime = feat_shape[1], feat_shape[2]

            self.encoders[sensor] = TemporalEncoder(
                in_channels=c_out,
                d_z=d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            self.decoders[sensor] = FeatureDecoder(
                out_channels=c_out,
                seq_len=t_prime,
                d_z=d_z,
                d_model=config.d_model,
            )

            n_classes = len(CLASS_MAP)
            self.pres_heads[sensor] = LinearPresenceHead(d_in=CausalLatentSpace.D_PRES)
            self.type_heads[sensor] = LinearTypeHead(d_in=CausalLatentSpace.D_TYPE, n_classes=n_classes)
            self.prox_heads[sensor] = LinearProximityHead(d_in=CausalLatentSpace.D_PROX)

            # Auxiliary heads (pre-training only)
            self.aux_pres_heads[sensor] = nn.Linear(CausalLatentSpace.D_PRES, 1)
            self.aux_type_heads[sensor] = nn.Linear(CausalLatentSpace.D_TYPE, n_classes)

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
        """All parameters except downstream heads and aux heads."""
        exclude = set(
            list(self.pres_heads.parameters()) +
            list(self.type_heads.parameters()) +
            list(self.prox_heads.parameters()) +
            list(self.aux_pres_heads.parameters()) +
            list(self.aux_type_heads.parameters())
        )
        return [p for p in self.parameters() if p not in exclude]

    def head_parameters(self):
        return (
            list(self.pres_heads.parameters()) +
            list(self.type_heads.parameters()) +
            list(self.prox_heads.parameters())
        )


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
        self.best_ref_elbo  = float("inf")
        self.patience_ctr   = 0
        self.scaler         = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        # Adaptive beta state (Options 1 + 3)
        self.beta           = 0.0
        self.prev_val_recon = float("inf")

    # ------------------------------------------------------------------
    # CRL pre-training
    # ------------------------------------------------------------------

    def train_crl(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        fieldnames = [
            "epoch", "beta", "beta_event", "lr", "grad_norm",
            "train_recon", "train_kl", "train_interv", "train_total",
            "train_aux_pres", "train_aux_type", "train_aux_prox",
            "val_recon",   "val_kl",   "val_interv",   "val_total",
            "val_aux_pres", "val_aux_type", "val_aux_prox",
            "val_raw_kl", "val_ref_elbo",
            "pres_changed_frac", "type_changed_frac",
        ]
        metrics_path = self.save_dir / "crl_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore').writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            train_m, grad_norm = self._train_epoch(train_loader, self.beta)
            val_m   = self._eval_epoch(val_loader, self.beta)
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]

            raw_kl          = val_m["val_raw_kl"]
            recon_improving = val_m["val_recon"] < self.prev_val_recon - self.cfg.recon_min_delta
            self.prev_val_recon = val_m["val_recon"]

            if raw_kl < self.cfg.kl_floor:
                self.beta = max(0.0, self.beta - self.cfg.beta_step)
                beta_event = "↓collapse"
            elif recon_improving or raw_kl > self.cfg.kl_target:
                self.beta = min(1.0, self.beta + self.cfg.beta_step)
                beta_event = "↑"
            else:
                beta_event = "→hold"

            row = {
                "epoch": epoch, "beta": round(self.beta, 4), "beta_event": beta_event,
                "lr": round(current_lr, 6), "grad_norm": round(grad_norm, 4),
                **train_m, **val_m,
                "pres_changed_frac": val_m.get("val_pres_changed_frac", 0.0),
                "type_changed_frac": val_m.get("val_type_changed_frac", 0.0),
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore').writerow(row)

            print(
                f"Epoch {epoch:3d} | β={self.beta:.2f}({beta_event}) lr={current_lr:.2e} | "
                f"recon={val_m['val_recon']:.3f} kl={val_m['val_raw_kl']:.3f} "
                f"interv={val_m['val_interv']:.3f} | "
                f"aux_p={val_m['val_aux_pres']:.3f} aux_t={val_m['val_aux_type']:.3f} "
                f"aux_px={val_m['val_aux_prox']:.3f} | "
                f"ref_elbo={val_m['val_ref_elbo']:.3f} | "
                f"‖g‖={grad_norm:.2f} | "
                f"Δpres={val_m['val_pres_changed_frac']:.2f} Δtype={val_m['val_type_changed_frac']:.2f}"
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
        """
        Shared forward pass for train and eval.
        Returns (loss, metrics_dict).
        """
        recon_total    = torch.tensor(0.0, device=self.device)
        kl_total       = torch.tensor(0.0, device=self.device)
        interv_total   = torch.tensor(0.0, device=self.device)
        aux_pres_total = torch.tensor(0.0, device=self.device)
        aux_type_total = torch.tensor(0.0, device=self.device)
        aux_prox_total = torch.tensor(0.0, device=self.device)
        raw_kl_total = 0.0
        pres_changed_total = type_changed_total = 0.0
        n_mod = 0

        use_aux = self.cfg.use_aux_supervision
        use_label_change = (self.cfg.intervention_mode == "label_change")

        for sensor in self.model.sensors:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue

            x_t  = batch[f"x_{sensor}_t"][avail].to(self.device)
            x_tn = batch[f"x_{sensor}_tn"][avail].to(self.device)

            x_both = torch.cat([x_t, x_tn], dim=0)
            feat_both, z_both, mu_both, lv_both = self.model.encode(sensor, x_both)
            B = x_t.shape[0]
            feat_t, feat_tn = feat_both[:B], feat_both[B:]
            z_t,   z_tn    = z_both[:B],    z_both[B:]
            mu_t,  mu_tn   = mu_both[:B],   mu_both[B:]
            lv_t,  lv_tn   = lv_both[:B],   lv_both[B:]

            assert mu_both.isfinite().all(), f"mu non-finite for {sensor}"
            assert lv_both.isfinite().all(), f"lv non-finite for {sensor}"

            x_hat_both = self.model.decode(sensor, z_both)
            x_hat_t, x_hat_tn = x_hat_both[:B], x_hat_both[B:]

            recon = (reconstruction_loss(x_hat_t, feat_t) +
                     reconstruction_loss(x_hat_tn, feat_tn)) / 2

            kl_t  = kl_divergence(mu_t,  lv_t,  beta=beta)
            kl_tn = kl_divergence(mu_tn, lv_tn, beta=beta)
            kl    = (kl_t + kl_tn) / 2

            raw_kl = (kl_divergence(mu_t,  lv_t,  beta=1.0) +
                      kl_divergence(mu_tn, lv_tn, beta=1.0)) / 2
            raw_kl_total += raw_kl.item()

            # Split latent into named subspaces
            z_pres_t, z_type_t, z_prox_t, z_env_t, _ = self.model.latent.split(z_t)
            z_pres_tn, z_type_tn, z_prox_tn, z_env_tn, _ = self.model.latent.split(z_tn)

            # Intervention loss
            if use_label_change:
                det_t_lbl  = batch["detection_label_t"][avail].to(self.device)
                det_tn_lbl = batch["detection_label_tn"][avail].to(self.device)
                typ_t_lbl  = batch["vehicle_type_t"][avail].to(self.device)
                typ_tn_lbl = batch["vehicle_type_tn"][avail].to(self.device)
                targets = label_change_target(det_t_lbl, det_tn_lbl, typ_t_lbl, typ_tn_lbl)
                logits  = self.model.interv_classifier(z_env_t, z_env_tn)
                interv  = F.binary_cross_entropy_with_logits(logits, targets)
                pres_changed_total += targets[:, 0].mean().item()
                type_changed_total += targets[:, 1].mean().item()
            else:
                interv = torch.tensor(0.0, device=self.device)

            # Auxiliary supervision losses
            aux_pres = aux_type = aux_prox = torch.tensor(0.0, device=self.device)
            if use_aux:
                det_lbl = batch["detection_label"][avail].to(self.device)
                typ_lbl = batch["vehicle_type"][avail].to(self.device)

                for z_pres_x in [z_pres_t, z_pres_tn]:
                    logit = self.model.aux_pres_heads[sensor](z_pres_x).squeeze(-1)
                    aux_pres = aux_pres + F.binary_cross_entropy_with_logits(
                        logit, det_lbl.float()
                    ) / 2

                mask = (det_lbl == 1) & (typ_lbl >= 0)
                if mask.any():
                    for z_type_x in [z_type_t, z_type_tn]:
                        logits_type = self.model.aux_type_heads[sensor](z_type_x[mask])
                        aux_type = aux_type + F.cross_entropy(
                            logits_type, typ_lbl[mask]
                        ) / 2

                for x_raw, z_prox_x in [(x_t, z_prox_t), (x_tn, z_prox_tn)]:
                    rms = x_raw.pow(2).mean(dim=-1).sqrt().mean(dim=-1)
                    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
                    prox_pred = z_prox_x.mean(dim=-1)
                    aux_prox = aux_prox + F.mse_loss(prox_pred, rms_norm) / 2

            recon_total    += recon
            kl_total       += kl
            interv_total   += interv
            aux_pres_total += aux_pres
            aux_type_total += aux_type
            aux_prox_total += aux_prox
            n_mod += 1

        if n_mod == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {
                "recon": 0.0, "kl": 0.0, "raw_kl": 0.0, "interv": 0.0, "total": 0.0,
                "aux_pres": 0.0, "aux_type": 0.0, "aux_prox": 0.0,
                "pres_changed_frac": 0.0, "type_changed_frac": 0.0,
            }

        recon_total    = recon_total    / n_mod
        kl_total       = kl_total       / n_mod
        raw_kl_total   = raw_kl_total   / n_mod
        interv_total   = interv_total   / n_mod
        aux_pres_total = aux_pres_total / n_mod
        aux_type_total = aux_type_total / n_mod
        aux_prox_total = aux_prox_total / n_mod

        total = (recon_total + kl_total
                 + self.cfg.lambda_interv  * interv_total
                 + self.cfg.lambda_aux_pres * aux_pres_total
                 + self.cfg.lambda_aux_type * aux_type_total
                 + self.cfg.lambda_aux_prox * aux_prox_total)

        return total, {
            "recon":  recon_total.item(),
            "kl":     kl_total.item(),
            "raw_kl": raw_kl_total,
            "interv": interv_total.item(),
            "total":  total.item(),
            "aux_pres": aux_pres_total.item(),
            "aux_type": aux_type_total.item(),
            "aux_prox": aux_prox_total.item(),
            "pres_changed_frac": pres_changed_total / n_mod,
            "type_changed_frac": type_changed_total / n_mod,
        }

    def _train_epoch(self, loader: DataLoader, beta: float) -> tuple[dict, float]:
        self.model.train()
        accum: dict[str, float] = {}
        n = 0
        total_grad_norm = 0.0
        for batch in loader:
            for sensor in self.model.sensors:
                if batch[f"{sensor}_avail"].any():
                    assert batch[f"x_{sensor}_t"].isfinite().all()
                    assert batch[f"x_{sensor}_tn"].isfinite().all()

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss, metrics = self._forward_pair(batch, beta)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            if self.cfg.steps_per_epoch and n >= self.cfg.steps_per_epoch:
                break
        avg_grad_norm = total_grad_norm / max(n, 1)
        result = {
            f"train_{k}": v / max(n, 1)
            for k, v in accum.items()
            if k not in ("raw_kl", "pres_changed_frac", "type_changed_frac")
        }
        return result, avg_grad_norm

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.eval()
        accum: dict[str, float] = {}
        n = 0
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                _, metrics = self._forward_pair(batch, beta)
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
        self.model.train()
        out = {f"val_{k}": v / max(n, 1) for k, v in accum.items()}
        out["val_ref_elbo"] = (
            out["val_recon"] + out["val_raw_kl"] +
            self.cfg.lambda_interv * out["val_interv"]
        )
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
                      "val_pres_acc", "val_pres_f1", "val_type_acc", "val_type_f1",
                      "val_prox_loss", "class_breakdown"]
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
                z_pres, z_type, _, _, _ = self.model.latent.split(z)

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
        import json as _json

        self.model.eval()
        pres_head = self.model.pres_heads[sensor]
        type_head = self.model.type_heads[sensor]
        prox_head = self.model.prox_heads[sensor]

        det_true, det_pred = [], []
        cls_true, cls_pred = [], []
        prox_losses = []

        for batch in loader:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue
            x = batch[f"x_{sensor}"][avail].to(self.device)
            det   = batch["detection_label"][avail]
            vtype = batch["vehicle_type"][avail]

            _, z, _, _ = self.model.encode(sensor, x)
            z_pres, z_type, z_prox, _, _ = self.model.latent.split(z)

            pres_logit  = pres_head(z_pres).squeeze(-1)
            type_logits = type_head(z_type)
            prox_pred   = prox_head(z_prox).squeeze(-1)

            rms = x.pow(2).mean(dim=-1).sqrt().mean(dim=-1)
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
            prox_losses.append(F.mse_loss(prox_pred, rms_norm).item())

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
        val_prox_loss = sum(prox_losses) / max(len(prox_losses), 1)

        if cls_true:
            per_class = f1_score(cls_true, cls_pred, average=None, zero_division=0)
            from crl_vehicle.config import CLASS_MAP
            class_breakdown = _json.dumps({CLASS_MAP.get(i, str(i)): round(float(v), 4)
                                           for i, v in enumerate(per_class)})
        else:
            class_breakdown = "{}"

        self.model.train()
        return {
            "val_pres_acc":    pres_acc,
            "val_pres_f1":     pres_f1,
            "val_type_acc":    type_acc,
            "val_type_f1":     type_f1,
            "val_prox_loss":   val_prox_loss,
            "class_breakdown": class_breakdown,
        }
