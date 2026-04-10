"""
Trainer

Orchestrates the full CRL training pipeline for independent audio and
seismic modalities.  Each modality has its own:
    - SpectrogramFrontend
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
from crl_vehicle.models.spectrogram import SpectrogramFrontend
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
from crl_vehicle.data.transforms import apply_interventions_batch_gpu, N_INTERVENTIONS
from training.scheduler import build_scheduler
from training.eval import sample_level_eval, plot_confusion_matrices


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
            self.filterbanks[sensor] = SpectrogramFrontend(mod_cfg)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run Filterbank → SSM → CausalEncoder for one modality.
        Returns (z, mu, log_var, y) where y is the filterbank output (B, K*C, T').
        Returning y avoids a redundant filterbank forward pass when the caller
        also needs the filterbank output as a reconstruction target.
        """
        y = self.filterbanks[sensor](x)  # (B, K*C, T')
        h = self.ssms[sensor](y)         # (B, T', d_model)
        z, mu, log_var = self.encoders[sensor](h)
        return z, mu, log_var, y

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

            z, mu, log_var, y = self.encode_modality(sensor, x)

            # Detach spectrogram output for use as reconstruction target —
            # stop-gradient so the target doesn't move with encoder updates.
            mod_cfg = self.cfg.modality_cfg(sensor)
            K = mod_cfg.filterbank_out_channels
            T = mod_cfg.t_prime
            x_target = y.detach().view(x.shape[0], K, T)

            z_scm = self.scm(z, interv_mask)
            x_hat = self.decoders[sensor](z)

            outputs[f"x_raw_{sensor}"] = x          # cached for contrast loss (already on device)
            outputs[f"z_{sensor}"] = z
            outputs[f"mu_{sensor}"] = mu
            outputs[f"log_var_{sensor}"] = log_var
            outputs[f"z_scm_{sensor}"] = z_scm
            outputs[f"x_hat_{sensor}"] = x_hat
            outputs[f"x_target_{sensor}"] = x_target

        outputs["scm_l1"] = self.scm.adjacency().abs().sum()

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

    def forward_horizon_pair(self, batch: dict, device: torch.device) -> dict:
        """
        Forward pass for multi-horizon pair batches (MultiHorizonPairDataset).
        Encodes x_t and x_tn per modality; runs the intervention classifier on
        the temporal difference; stores z_tn for temporal consistency loss.
        """
        outputs = {
            "interv_mask": None,
            "vehicle_labels": batch["vehicle_type"].to(device),
            "det_labels": batch["detection_label"].to(device),
            "horizon_n": batch["horizon_n"].to(device),
        }

        for sensor in self.sensors:
            x_t  = batch[f"x_{sensor}_t"].to(device)
            x_tn = batch[f"x_{sensor}_tn"].to(device)
            avail = batch[f"{sensor}_avail"].to(device)
            outputs[f"avail_{sensor}"] = avail

            z_t,  mu_t,  lv_t,  y_t = self.encode_modality(sensor, x_t)
            z_tn, _, _, _            = self.encode_modality(sensor, x_tn)

            mod_cfg = self.cfg.modality_cfg(sensor)
            K = mod_cfg.filterbank_out_channels
            T = mod_cfg.t_prime
            x_target = y_t.detach().view(x_t.shape[0], K, T)

            z_scm = self.scm(z_t, intervention_mask=None)
            x_hat = self.decoders[sensor](z_t)

            outputs[f"z_{sensor}"]       = z_t
            outputs[f"z_tn_{sensor}"]    = z_tn
            outputs[f"mu_{sensor}"]      = mu_t
            outputs[f"log_var_{sensor}"] = lv_t
            outputs[f"z_scm_{sensor}"]   = z_scm
            outputs[f"x_hat_{sensor}"]   = x_hat
            outputs[f"x_target_{sensor}"] = x_target

            # Unknown intervention classifier (once, on primary sensor)
            if "interv_logits" not in outputs:
                interv_logits = self.unknown_interv(z_t, z_tn)
                outputs["interv_logits"] = interv_logits
                interv_t  = batch["interv_idx_t"].to(device)
                interv_tn = batch["interv_idx_tn"].to(device)
                noise_start = (
                    self.cfg.d_z_presence + self.cfg.d_z_type + self.cfg.d_z_proximity
                )
                targets = torch.zeros(z_t.shape[0], dtype=torch.long, device=device)
                changed = interv_t != interv_tn
                noise_dim = (interv_t[changed] - 1) % self.cfg.d_z_noise
                targets[changed] = noise_start + noise_dim + 1
                outputs["interv_targets"] = targets

        outputs["scm_l1"] = self.scm.adjacency().abs().sum()
        return outputs

    @torch.no_grad()
    def predict_with_vote(
        self,
        x_audio: torch.Tensor | None,
        x_seismic: torch.Tensor | None,
        weights: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Independent per-modality inference with weighted vote.

        Each modality produces its own detection probability and vehicle-type
        probability.  Results are combined by a weighted average, enabling
        graceful degradation when one modality is unavailable.

        weights: {sensor: float} — default equal weights.
                 Calibrate post-training with calibrate_vote_weights().
        Returns:
            det_prob  : (B,) probability of vehicle present
            type_probs: (B, n_classes) softmax probability
        """
        if weights is None:
            weights = {s: 1.0 for s in self.sensors}

        det_probs, type_probs, ws = [], [], []
        for sensor in self.sensors:
            x = x_audio if sensor == "audio" else x_seismic
            if x is None:
                continue
            z, _, _, _ = self.encode_modality(sensor, x)
            z_pres, z_type, _, _ = self.encoders[sensor].split_z_raw(z)
            pres_logit, type_logit = self.det_heads[sensor](z_pres, z_type)
            det_probs.append(torch.sigmoid(pres_logit) * weights[sensor])
            type_probs.append(torch.softmax(type_logit, dim=-1) * weights[sensor])
            ws.append(weights[sensor])

        if not det_probs:
            raise ValueError("No modality available for inference")

        W = sum(ws)
        return sum(det_probs) / W, sum(type_probs) / W

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
            [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "det_heads" not in n and n != "scm.A_raw"
                    ],
                    "weight_decay": config.wd,
                },
                {
                    "params": [model.scm.A_raw],
                    "weight_decay": 0.0,
                },
            ],
            lr=config.lr,
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
            "scm_l1",
            "beta",
            "l1_w",
        ]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            self.loss_fn.update_beta(epoch)
            self.loss_fn.update_lambda_l1(epoch)
            train_metrics = self._train_epoch(loader_known, loader_pairs, epoch)
            # Single val pass: annealing beta (logging) + fixed beta (checkpointing).
            # Fixed-beta metric is epoch-invariant and used for model selection.
            both = self._eval_crl(
                val_loader,
                {"annealing": None, "fixed": self.cfg.beta_end},
            )
            val_metrics = both["annealing"]
            val_ckpt_metrics = both["fixed"]
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
                f"scm_l1={val_metrics.get('scm_l1', 0):.3f} "
                f"β={self.loss_fn.current_beta:.2f} "
                f"λ_l1={self.loss_fn.current_lambda_l1:.4f}"
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

            # All-interventions contrast loss: expand batch to (N+1) intervention
            # views in one vectorised GPU call, encode all, then penalise vehicle-dim
            # variance across interventions and reward noise-dim variance.
            # Subsample to min(B, 16) to keep memory bounded.
            if self.cfg.lambda_contrast > 0:
                for sensor in self.model.sensors:
                    avail = outputs.get(f"avail_{sensor}")
                    x_raw = outputs[f"x_raw_{sensor}"]   # already on device, no .to() needed
                    if avail is not None:
                        x_raw = x_raw[avail]
                    if x_raw.shape[0] == 0:
                        continue
                    sub_B = min(x_raw.shape[0], 16)
                    if sub_B < x_raw.shape[0]:
                        idx = torch.randperm(x_raw.shape[0], device=x_raw.device)[:sub_B]
                        x_raw = x_raw[idx]
                    sr = self.cfg.modality_cfg(sensor).sample_rate
                    N = N_INTERVENTIONS
                    # (N+1, sub_B, C, W) — all views generated on GPU in one call
                    all_views = apply_interventions_batch_gpu(x_raw, sr)
                    views = all_views.view((N + 1) * sub_B, *x_raw.shape[1:])
                    z_all, _, _, _ = self.model.encode_modality(sensor, views)
                    z_all = z_all.view(N + 1, sub_B, -1).permute(1, 0, 2)  # (sub_B, N+1, d_z)
                    enc = self.model.encoders[sensor]
                    noise_start = enc.noise_idx.start
                    z_veh = z_all[:, :, :noise_start]          # (sub_B, N+1, d_veh)
                    z_nz  = z_all[:, :, enc.noise_idx]         # (sub_B, N+1, d_noise)
                    L_inv = (z_veh - z_veh.mean(dim=1, keepdim=True)).pow(2).mean()
                    L_equiv = -z_nz.var(dim=1).clamp(max=1.0).mean()
                    loss = (
                        loss
                        + self.cfg.lambda_contrast * L_inv
                        + self.cfg.lambda_equiv * L_equiv
                    )
                    metrics[f"contrast_inv_{sensor}"] = L_inv.item()
                    metrics[f"contrast_equiv_{sensor}"] = L_equiv.item()

            # Horizon pair pass (curriculum)
            if pair_iter is not None:
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(loader_pairs)
                    pair_batch = next(pair_iter)
                pair_outputs = self.model.forward_horizon_pair(pair_batch, self.device)
                pair_loss, _ = self.loss_fn(pair_outputs)
                loss = loss + unknown_weight * pair_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1
            if self.cfg.steps_per_epoch and n_batches >= self.cfg.steps_per_epoch:
                break

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_crl(
        self,
        loader: DataLoader,
        betas: dict[str, float | None] | None = None,
    ) -> dict[str, dict]:
        """
        Single-pass validation over loader, computing metrics at each beta in
        `betas`.  Returns {key: metrics_dict} so callers can retrieve annealing
        and fixed-beta metrics without traversing the val set twice.
        """
        if betas is None:
            betas = {"default": None}
        self.model.eval()
        totals: dict[str, dict] = {key: {} for key in betas}
        n = 0
        for batch in loader:
            outputs = self.model.forward_known(batch, self.device)
            for key, beta in betas.items():
                _, metrics = self.loss_fn(outputs, beta_override=beta)
                for k, v in metrics.items():
                    totals[key][k] = totals[key].get(k, 0.0) + v
            n += 1
        self.model.train()
        return {
            key: {k: v / max(n, 1) for k, v in m.items()}
            for key, m in totals.items()
        }

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
                    _, mu, _, _ = self.model.encode_modality(ref, x_ref)

                enc = self.model.encoders[ref]
                z_pres, z_type, _, _ = enc.split_z_raw(mu)

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

            train_det_loss = epoch_det / max(n_det, 1)
            train_cls_loss = epoch_cls / max(n_cls, 1)

            # Gradient norm on head parameters (diagnostic: zero = no learning)
            head_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.model.head_parameters()
                if p.grad is not None
            ) ** 0.5

            row = {
                "epoch": epoch,
                "train_det_loss": train_det_loss,
                "train_cls_loss": train_cls_loss,
                **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | "
                f"tr_det={train_det_loss:.4f} tr_cls={train_cls_loss:.4f} "
                f"det_f1={val_m['val_det_f1']:.4f} cls_f1={val_m['val_cls_f1']:.4f} "
                f"grad={head_grad_norm:.2e}"
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

        # --- Post-training diagnostics: confusion matrices + per-sample CSV ---
        ref = "seismic" if "seismic" in self.model.sensors else self.model.sensors[0]
        best_head_path = self.save_dir / f"det_head_{ref}_best.pth"
        if best_head_path.exists():
            self.model.det_heads[ref].load_state_dict(
                torch.load(best_head_path, map_location=self.device)
            )
        diag_dir = self.save_dir / "diagnostics"
        print(f"  Generating confusion matrices → {diag_dir}/")
        df_preds = sample_level_eval(self.model, val_loader, self.device, primary_sensor=ref)
        plot_confusion_matrices(df_preds, diag_dir)
        print(f"  Saved: detection_cm.png, type_cm.png, det_acc_by_interv.png, predictions.csv")

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

            _, mu, _, _ = self.model.encode_modality(ref, x_ref)
            enc = self.model.encoders[ref]
            z_pres, z_type, _, _ = enc.split_z_raw(mu)
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

    @torch.no_grad()
    def calibrate_vote_weights(self, val_loader: DataLoader) -> dict:
        """
        Compute per-modality detection AUC on the validation set and return
        normalised vote weights proportional to (AUC - 0.5).

        Call after train_downstream() to produce calibrated weights for
        predict_with_vote().  Modalities near random (AUC ≈ 0.5) receive low
        weight; strong modalities receive high weight.

        Returns: {sensor: weight} with weights summing to 1.0.
        """
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        aucs = {}
        for sensor in self.model.sensors:
            scores, labels = [], []
            for batch in val_loader:
                x = batch[f"x_{sensor}"].to(self.device)
                z, _, _, _ = self.model.encode_modality(sensor, x)
                z_pres, z_type, _, _ = self.model.encoders[sensor].split_z_raw(z)
                pres_logit, _ = self.model.det_heads[sensor](z_pres, z_type)
                scores.extend(torch.sigmoid(pres_logit).cpu().tolist())
                labels.extend(batch["detection_label"].tolist())
            try:
                auc = roc_auc_score(labels, scores)
            except Exception:
                auc = 0.5
            # Weight = excess above chance; floor at 0.1 so no modality is silenced
            aucs[sensor] = float(max(0.1, auc - 0.5))

        total = sum(aucs.values())
        weights = {s: v / total for s, v in aucs.items()}
        self.model.train()
        print(f"Vote weights (AUC-based): {weights}")
        return weights
