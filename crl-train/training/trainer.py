"""
Trainer

Orchestrates the full CRL training pipeline for independent audio and
seismic modalities.  Each modality has its own:
    - LearnableFilterbank
    - TemporalSSM
    - CausalEncoder
    - SpectralDecoder
    - VehicleDetectionHead

One SCM is shared across modalities (same causal graph for both sensors).
One UnknownInterventionClassifier is shared.

Training curriculum (from the plan):
    Epochs 0 – unknown_interv_start_epoch:
        Known interventions only.
    Epochs unknown_interv_start_epoch – +unknown_interv_ramp_epochs:
        Linear ramp of unknown-intervention pair batches (0→50%).
    Epochs beyond:
        50/50 known + unknown.

Downstream phase:
    Freeze all CRL modules; train VehicleDetectionHead only.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.models.filterbank import LearnableFilterbank
from crl_vehicle.models.ssm import TemporalSSM
from crl_vehicle.models.encoder import CausalEncoder
from crl_vehicle.models.scm import SCM
from crl_vehicle.models.intervention import (
    KnownInterventionHandler,
    UnknownInterventionClassifier,
)
from crl_vehicle.models.decoder import SpectralDecoder
from crl_vehicle.models.downstream import VehicleDetectionHead
from crl_vehicle.losses.combined import CombinedLoss
from training.scheduler import build_scheduler


# ---------------------------------------------------------------------------
# CRLModel: assembles per-modality + shared components
# ---------------------------------------------------------------------------


class CRLModel(nn.Module):
    """
    Full CRL model: one pipeline per modality, shared SCM and intervention
    classifier.

    Args:
        config  : CRLConfig
        sensors : list of sensor names to build (subset of MODALITIES)
    """

    def __init__(self, config: CRLConfig, sensors: list | None = None):
        super().__init__()
        self.cfg = config
        self.sensors = sensors or MODALITIES

        # Per-modality components
        self.filterbanks = nn.ModuleDict()
        self.ssms = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for sensor in self.sensors:
            mod_cfg = config.modality_cfg(sensor)
            self.filterbanks[sensor] = LearnableFilterbank(mod_cfg, sensor=sensor)
            self.ssms[sensor] = TemporalSSM(
                in_channels=mod_cfg.filterbank_out_channels,
                config=config,
            )
            self.encoders[sensor] = CausalEncoder(
                d_model=config.d_model,
                d_z_presence=config.d_z_presence,
                d_z_type=config.d_z_type,
                d_z_proximity=config.d_z_proximity,
                d_z_noise=config.d_z_noise,
            )
            self.decoders[sensor] = SpectralDecoder(
                d_z=config.d_z,
                d_model=config.d_model,
                mod_cfg=mod_cfg,
            )

        # Shared SCM (one causal graph for all sensors)
        self.scm = SCM(d_z=config.d_z, hidden_dim=config.scm_hidden)

        # Intervention modules
        noise_start = config.d_z_presence + config.d_z_type + config.d_z_proximity
        self.known_interv = KnownInterventionHandler(
            d_z=config.d_z, noise_start_idx=noise_start
        )
        self.unknown_interv = UnknownInterventionClassifier(d_z=config.d_z)

        # Downstream heads (one per modality)
        from crl_vehicle.config import CLASS_MAP

        n_classes = len(CLASS_MAP)
        self.det_heads = nn.ModuleDict(
            {
                sensor: VehicleDetectionHead(
                    d_z_presence=config.d_z_presence,
                    d_z_type=config.d_z_type,
                    n_classes=n_classes,
                )
                for sensor in self.sensors
            }
        )

    def encode_modality(
        self, sensor: str, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run Filterbank → SSM → CausalEncoder for one modality.
        Returns (z, mu, log_var) each (B, d_z).
        """
        y = self.filterbanks[sensor](x)  # (B, K*C, T')
        h = self.ssms[sensor](y)  # (B, T', d_model)
        return self.encoders[sensor](h)  # z, mu, log_var

    def forward_known(self, batch: dict, device: torch.device) -> dict:
        """
        Forward pass for known-intervention batches.
        Returns outputs dict for CombinedLoss.
        """
        interv_idx = batch["interv_idx"].to(device)
        interv_mask = self.known_interv(interv_idx)  # (B, d_z)

        outputs = {
            "interv_mask": interv_mask,
            "vehicle_labels": batch["vehicle_type"].to(device),
            "det_labels": batch["detection_label"].to(device),
        }

        for sensor in self.sensors:
            key = f"x_{sensor}"
            avail_key = f"{sensor}_avail"
            x = batch[key].to(device)
            avail = batch[avail_key].to(device)
            outputs[f"avail_{sensor}"] = avail

            z, mu, log_var = self.encode_modality(sensor, x)

            # Filterbank target (stop gradient — we're reconstructing the
            # actual filterbank output, not re-computing through FB)
            with torch.no_grad():
                fb_target = self.filterbanks[sensor](x)  # (B, K*C, T')
            # Reshape to (B, K, T') for reconstruction loss
            mod_cfg = self.cfg.modality_cfg(sensor)
            K = mod_cfg.n_filters
            T = mod_cfg.t_prime
            x_target = fb_target.view(x.shape[0], K, T)

            z_scm = self.scm(z, interv_mask)
            x_hat = self.decoders[sensor](z)

            outputs[f"z_{sensor}"] = z
            outputs[f"mu_{sensor}"] = mu
            outputs[f"log_var_{sensor}"] = log_var
            outputs[f"z_scm_{sensor}"] = z_scm
            outputs[f"x_hat_{sensor}"] = x_hat
            outputs[f"x_target_{sensor}"] = x_target

        # Acyclicity loss from shared SCM
        outputs["acyclicity"] = self.scm.acyclicity_loss()

        # Downstream logits (using seismic if available, else audio)
        ref_sensor = "seismic" if "seismic" in self.sensors else self.sensors[0]
        z_ref = outputs[f"z_{ref_sensor}"]
        enc = self.encoders[ref_sensor]
        z_pres, z_type, _, _ = enc.split_z_raw(z_ref)
        pres_logit, type_logits = self.det_heads[ref_sensor](z_pres, z_type)
        # Convert binary presence to 2-class logits for CrossEntropyLoss
        outputs["det_logits"] = torch.stack([-pres_logit, pres_logit], dim=1)
        outputs["vehicle_logits"] = type_logits

        return outputs

    def forward_unknown(self, batch: dict, device: torch.device) -> dict:
        """
        Forward pass for unknown-intervention (consecutive pair) batches.
        Returns outputs dict including intervention classifier logits.
        """
        outputs = {
            "interv_mask": None,
            "vehicle_labels": batch["vehicle_type"].to(device),
            "det_labels": batch["detection_label"].to(device),
        }

        for sensor in self.sensors:
            x_t = batch[f"x_{sensor}_t"].to(device)
            x_t1 = batch[f"x_{sensor}_t1"].to(device)
            avail = batch[f"{sensor}_avail"].to(device)
            outputs[f"avail_{sensor}"] = avail

            z_t, mu_t, lv_t = self.encode_modality(sensor, x_t)
            z_t1, _, _ = self.encode_modality(sensor, x_t1)

            mod_cfg = self.cfg.modality_cfg(sensor)
            K = mod_cfg.n_filters
            T = mod_cfg.t_prime
            with torch.no_grad():
                fb_t = self.filterbanks[sensor](x_t)
            x_target = fb_t.view(x_t.shape[0], K, T)

            z_scm = self.scm(z_t, intervention_mask=None)
            x_hat = self.decoders[sensor](z_t)

            outputs[f"z_{sensor}"] = z_t
            outputs[f"mu_{sensor}"] = mu_t
            outputs[f"log_var_{sensor}"] = lv_t
            outputs[f"z_scm_{sensor}"] = z_scm
            outputs[f"x_hat_{sensor}"] = x_hat
            outputs[f"x_target_{sensor}"] = x_target

            # Unknown intervention classification (once, using primary sensor)
            if "interv_logits" not in outputs:
                interv_logits = self.unknown_interv(z_t, z_t1)
                outputs["interv_logits"] = interv_logits
                # No ground-truth for unknown — this trains the classifier
                # via a pseudo-label derived from the intervention indices.
                interv_t = batch["interv_idx_t"].to(device)
                interv_t1 = batch["interv_idx_t1"].to(device)
                # Target: index of the noise dimension that changed.
                # When interv_t != interv_t1, target = noise_dim; else = 0.
                noise_start = (
                    self.cfg.d_z_presence + self.cfg.d_z_type + self.cfg.d_z_proximity
                )
                targets = torch.zeros(z_t.shape[0], dtype=torch.long, device=device)
                changed = interv_t != interv_t1
                noise_dim = (interv_t[changed] - 1) % self.cfg.d_z_noise
                targets[changed] = (
                    noise_start + noise_dim + 1
                )  # +1 for "no-interv" class 0
                outputs["interv_targets"] = targets

        outputs["acyclicity"] = self.scm.acyclicity_loss()
        return outputs

    def crl_parameters(self):
        """All parameters except downstream heads."""
        skip = set(self.det_heads.parameters())
        return [p for p in self.parameters() if p not in skip]

    def head_parameters(self):
        """Only downstream head parameters."""
        return list(self.det_heads.parameters())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:

    def __init__(
        self,
        model: CRLModel,
        loss_fn: CombinedLoss,
        config: CRLConfig,
        device: torch.device,
        save_dir: Path,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.crl_parameters(),
            lr=config.lr,
            weight_decay=config.wd,
        )
        self.scheduler = build_scheduler(self.optimizer, config)
        self.best_val_loss = float("inf")
        self.patience_ctr = 0

    # ------------------------------------------------------------------
    # CRL pre-training
    # ------------------------------------------------------------------

    def train_crl(
        self,
        loader_known: DataLoader,
        loader_pairs: DataLoader | None,
        val_loader: DataLoader,
        epochs: int,
    ):
        metrics_path = self.save_dir / "crl_metrics.csv"
        fieldnames = [
            "epoch",
            "train_total",
            "val_total",
            "val_ckpt",
            "recon_audio",
            "kl_audio",
            "recon_seismic",
            "kl_seismic",
            "causal_audio",
            "causal_seismic",
            "acyclic",
            "beta",
            "acyclic_w",
        ]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            self.loss_fn.update_beta(epoch)
            self.loss_fn.update_lambda_acyclic(epoch)
            train_metrics = self._train_epoch(loader_known, loader_pairs, epoch)
            # Annealing val loss — for logging only (non-stationary across epochs)
            val_metrics = self._eval_crl(val_loader)
            # Fixed-beta val loss — evaluated at beta_end so comparisons are
            # epoch-invariant; used for checkpointing and early stopping.
            val_ckpt_metrics = self._eval_crl(
                val_loader, beta_override=self.cfg.beta_end
            )
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_total": train_metrics.get("total", 0.0),
                "val_total": val_metrics.get("total", 0.0),
                "val_ckpt": val_ckpt_metrics.get("total", 0.0),
            }
            row.update({k: val_metrics.get(k, 0.0) for k in fieldnames[4:]})

            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | "
                f"train={train_metrics.get('total', 0):.4f} "
                f"val={val_metrics.get('total', 0):.4f} "
                f"val_ckpt={val_ckpt_metrics.get('total', 0):.4f} "
                f"acyclic={val_metrics.get('acyclic', 0):.3f} "
                f"β={self.loss_fn.current_beta:.2f} "
                f"λ_acyc={self.loss_fn.current_lambda_acyclic:.2f}"
            )

            val_ckpt = val_ckpt_metrics.get("total", float("inf"))
            in_warmup = epoch < self.cfg.beta_anneal_epochs
            if val_ckpt < self.best_val_loss:
                self.best_val_loss = val_ckpt
                self.patience_ctr = 0
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / "crl_best.pth",
                )
                print(f"  ✓ New best (val_ckpt={val_ckpt:.4f})")
            elif not in_warmup:
                self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")

    def _train_epoch(
        self,
        loader_known: DataLoader,
        loader_pairs: DataLoader | None,
        epoch: int,
    ) -> dict:
        self.model.train()
        total_metrics = {}
        n_batches = 0

        # Curriculum: ramp unknown-intervention weight
        start = self.cfg.unknown_interv_start_epoch
        ramp = self.cfg.unknown_interv_ramp_epochs
        if epoch < start:
            unknown_weight = 0.0
        elif epoch < start + ramp:
            unknown_weight = 0.5 * (epoch - start) / ramp
        else:
            unknown_weight = 0.5

        pair_iter = iter(loader_pairs) if loader_pairs and unknown_weight > 0 else None

        for batch in loader_known:
            self.optimizer.zero_grad()

            # Known intervention pass
            outputs = self.model.forward_known(batch, self.device)
            loss, metrics = self.loss_fn(outputs)

            # Unknown intervention pass (curriculum)
            if pair_iter is not None:
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(loader_pairs)
                    pair_batch = next(pair_iter)
                unk_outputs = self.model.forward_unknown(pair_batch, self.device)
                unk_loss, _ = self.loss_fn(unk_outputs)
                loss = loss + unknown_weight * unk_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_crl(self, loader: DataLoader, beta_override: float | None = None) -> dict:
        self.model.eval()
        total_metrics = {}
        n = 0
        for batch in loader:
            outputs = self.model.forward_known(batch, self.device)
            _, metrics = self.loss_fn(outputs, beta_override=beta_override)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n += 1
        self.model.train()
        return {k: v / max(n, 1) for k, v in total_metrics.items()}

    # ------------------------------------------------------------------
    # Downstream head training
    # ------------------------------------------------------------------

    def train_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        # Freeze CRL backbone
        for name, p in self.model.named_parameters():
            if "det_heads" not in name:
                p.requires_grad = False

        head_opt = torch.optim.AdamW(self.model.head_parameters(), lr=self.cfg.lr * 0.1)
        head_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            head_opt, factor=0.5, patience=3
        )

        metrics_path = self.save_dir / "downstream_metrics.csv"
        fieldnames = [
            "epoch",
            "train_det_loss",
            "train_cls_loss",
            "val_det_acc",
            "val_det_f1",
            "val_cls_acc",
            "val_cls_f1",
        ]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        best_f1 = 0.0
        patience_ctr = 0

        # Class weights for imbalanced detection (vehicles are minority)
        det_weight = torch.tensor([1.0, 5.0], device=self.device)

        print("\n=== Downstream Head Training ===")
        for epoch in range(epochs):
            self.model.eval()
            for mod in self.model.det_heads.values():
                mod.train()

            epoch_det = 0.0
            epoch_cls = 0.0
            n_det = n_cls = 0

            for batch in train_loader:
                x_s = (
                    batch["x_seismic"].to(self.device)
                    if "seismic" in self.model.sensors
                    else None
                )
                x_a = (
                    batch["x_audio"].to(self.device)
                    if "audio" in self.model.sensors
                    else None
                )
                vtype = batch["vehicle_type"].to(self.device)
                det = batch["detection_label"].to(self.device)

                with torch.no_grad():
                    # Use seismic if available, else audio
                    ref = "seismic" if x_s is not None else "audio"
                    x_ref = x_s if x_s is not None else x_a
                    z, mu, _ = self.model.encode_modality(ref, x_ref)

                enc = self.model.encoders[ref]
                z_pres, z_type, _, _ = enc.split_z_raw(z)

                head_opt.zero_grad()
                pres_logit, type_logits = self.model.det_heads[ref](z_pres, z_type)

                det_logits_2c = torch.stack([-pres_logit, pres_logit], dim=1)
                det_loss = F.cross_entropy(det_logits_2c, det, weight=det_weight)

                cls_mask = vtype >= 0
                cls_loss = torch.tensor(0.0, device=self.device)
                if cls_mask.any():
                    cls_loss = F.cross_entropy(type_logits[cls_mask], vtype[cls_mask])

                loss = det_loss + cls_loss
                loss.backward()
                head_opt.step()

                epoch_det += det_loss.item()
                epoch_cls += cls_loss.item() if cls_mask.any() else 0.0
                n_det += 1
                n_cls += int(cls_mask.any())

            val_m = self._eval_downstream(val_loader)
            head_sched.step(-val_m["val_cls_f1"])

            row = {
                "epoch": epoch,
                "train_det_loss": epoch_det / max(n_det, 1),
                "train_cls_loss": epoch_cls / max(n_cls, 1),
                **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | "
                f"det_f1={val_m['val_det_f1']:.4f} "
                f"cls_f1={val_m['val_cls_f1']:.4f}"
            )

            if val_m["val_cls_f1"] > best_f1:
                best_f1 = val_m["val_cls_f1"]
                patience_ctr = 0
                for sensor, head in self.model.det_heads.items():
                    torch.save(
                        head.state_dict(),
                        self.save_dir / f"det_head_{sensor}_best.pth",
                    )
                print(f"  ✓ New best cls F1={best_f1:.4f}")
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        print(f"Downstream complete. Best cls F1: {best_f1:.4f}")

    @torch.no_grad()
    def _eval_downstream(self, loader: DataLoader) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()
        det_true, det_pred = [], []
        cls_true, cls_pred = [], []

        ref = "seismic" if "seismic" in self.model.sensors else self.model.sensors[0]

        for batch in loader:
            x_ref = batch[f"x_{ref}"].to(self.device)
            vtype = batch["vehicle_type"]
            det = batch["detection_label"]

            z, _, _ = self.model.encode_modality(ref, x_ref)
            enc = self.model.encoders[ref]
            z_pres, z_type, _, _ = enc.split_z_raw(z)
            pres_logit, type_logits = self.model.det_heads[ref](z_pres, z_type)

            det_pred.extend((pres_logit > 0).long().cpu().tolist())
            det_true.extend(det.tolist())

            cls_mask = vtype >= 0
            if cls_mask.any():
                cls_pred.extend(
                    type_logits[cls_mask.to(self.device)].argmax(1).cpu().tolist()
                )
                cls_true.extend(vtype[cls_mask].tolist())

        def _f1(true, pred):
            if not true:
                return 0.0, 0.0
            return (
                accuracy_score(true, pred),
                f1_score(true, pred, average="weighted", zero_division=0),
            )

        det_acc, det_f1 = _f1(det_true, det_pred)
        cls_acc, cls_f1 = _f1(cls_true, cls_pred)

        self.model.train()
        return {
            "val_det_acc": det_acc,
            "val_det_f1": det_f1,
            "val_cls_acc": cls_acc,
            "val_cls_f1": cls_f1,
        }
