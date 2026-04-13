"""
Trainer

Orchestrates the supervised multi-task CRL training pipeline.
Each modality has its own:
    - SpectrogramFrontend
    - TemporalSSM
    - MultiTaskEncoder  (produces e_pres, e_type, e_inst)
    - CRLHeads          (lightweight CRL-phase classifier heads, discarded after)
    - SpectralDecoder   (reconstruction regularizer)
    - VehicleDetectionHead (downstream-phase heads, separately trained)

CRL pre-training phase:
    Supervised losses on three task-specific embeddings:
        BCE(presence) + CE(type) + CE(instance) + MSE(reconstruction)
    Interventions are applied as data augmentation only — no intervention
    loss term, no curriculum scheduling.

Downstream phase:
    CRL backbone frozen. Three VehicleDetectionHead heads trained on frozen
    embeddings using the same supervised losses with class-weighted CE.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES, CLASS_MAP, INSTANCE_MAP
from crl_vehicle.models.spectrogram import SpectrogramFrontend
from crl_vehicle.models.ssm import TemporalSSM
from crl_vehicle.models.encoder import MultiTaskEncoder, CRLHeads
from crl_vehicle.models.decoder import SpectralDecoder
from crl_vehicle.models.downstream import VehicleDetectionHead
from crl_vehicle.losses.combined import SupervisedMultiTaskLoss
from training.scheduler import build_scheduler
from training.eval import run_full_eval, sample_level_eval, plot_confusion_matrices


# ---------------------------------------------------------------------------
# CRLModel: assembles per-modality components
# ---------------------------------------------------------------------------


class CRLModel(nn.Module):
    """
    Full CRL model: one pipeline per modality.

    During CRL training, crl_heads are used for task supervision.
    During downstream training, det_heads are used and crl_heads are ignored.

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
        self.ssms        = nn.ModuleDict()
        self.encoders    = nn.ModuleDict()
        self.crl_heads   = nn.ModuleDict()
        self.decoders    = nn.ModuleDict()

        n_type = len(CLASS_MAP)
        n_inst = len(INSTANCE_MAP)

        for sensor in self.sensors:
            mod_cfg = config.modality_cfg(sensor)
            self.filterbanks[sensor] = SpectrogramFrontend(mod_cfg)
            self.ssms[sensor] = TemporalSSM(
                in_channels=mod_cfg.filterbank_out_channels,
                config=config,
            )
            self.encoders[sensor] = MultiTaskEncoder(
                d_model=config.d_model,
                d_pres=config.d_pres,
                d_type=config.d_type,
                d_inst=config.d_inst,
            )
            self.crl_heads[sensor] = CRLHeads(
                d_pres=config.d_pres,
                d_type=config.d_type,
                d_inst=config.d_inst,
                n_type=n_type,
                n_inst=n_inst,
            )
            self.decoders[sensor] = SpectralDecoder(
                d_z=config.d_embed,
                d_model=config.d_model,
                mod_cfg=mod_cfg,
            )

        # Downstream heads (one per modality, separately trained)
        self.det_heads = nn.ModuleDict(
            {
                sensor: VehicleDetectionHead(
                    d_pres=config.d_pres,
                    d_type=config.d_type,
                    d_inst=config.d_inst,
                    n_classes=n_type,
                    n_instance_classes=n_inst,
                )
                for sensor in self.sensors
            }
        )

    def encode_modality(
        self, sensor: str, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run Filterbank → SSM → MultiTaskEncoder for one modality.
        Returns (e_pres, e_type, e_inst, y) where y is the filterbank output.
        Returning y avoids a redundant filterbank forward pass for reconstruction.
        """
        y = self.filterbanks[sensor](x)        # (B, K*C, T')
        h = self.ssms[sensor](y)               # (B, T', d_model)
        e_pres, e_type, e_inst = self.encoders[sensor](h)
        return e_pres, e_type, e_inst, y

    def forward(self, batch: dict, device: torch.device) -> dict:
        """
        Forward pass for CRL training.
        Returns outputs dict for SupervisedMultiTaskLoss.
        """
        outputs = {
            "detection_label": batch["detection_label"].to(device),
            "vehicle_type":    batch["vehicle_type"].to(device),
            "instance_type":   batch["instance_type"].to(device),
        }

        for sensor in self.sensors:
            x     = batch[f"x_{sensor}"].to(device)
            avail = batch[f"{sensor}_avail"].to(device)
            outputs[f"avail_{sensor}"] = avail

            e_pres, e_type, e_inst, y = self.encode_modality(sensor, x)

            mod_cfg = self.cfg.modality_cfg(sensor)
            K = mod_cfg.filterbank_out_channels
            T = mod_cfg.t_prime
            x_target = y.detach().view(x.shape[0], K, T)

            # Concatenate embeddings for reconstruction
            e_cat = torch.cat([e_pres, e_type, e_inst], dim=-1)  # (B, d_embed)
            x_hat = self.decoders[sensor](e_cat)

            outputs[f"e_pres_{sensor}"]   = e_pres
            outputs[f"e_type_{sensor}"]   = e_type
            outputs[f"e_inst_{sensor}"]   = e_inst
            outputs[f"x_hat_{sensor}"]    = x_hat
            outputs[f"x_target_{sensor}"] = x_target

            # CRL task supervision logits
            pres_logit, type_logits, inst_logits = self.crl_heads[sensor](e_pres, e_type, e_inst)
            outputs[f"pres_logit_{sensor}"]  = pres_logit
            outputs[f"type_logits_{sensor}"] = type_logits
            outputs[f"inst_logits_{sensor}"] = inst_logits

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
        Returns (det_prob, type_probs): (B,) and (B, n_type).
        """
        if weights is None:
            weights = {s: 1.0 for s in self.sensors}

        det_probs, type_probs, ws = [], [], []
        for sensor in self.sensors:
            x = x_audio if sensor == "audio" else x_seismic
            if x is None:
                continue
            e_pres, e_type, e_inst, _ = self.encode_modality(sensor, x)
            pres_logit, type_logit, _ = self.det_heads[sensor](e_pres, e_type, e_inst)
            det_probs.append(torch.sigmoid(pres_logit) * weights[sensor])
            type_probs.append(torch.softmax(type_logit, dim=-1) * weights[sensor])
            ws.append(weights[sensor])

        if not det_probs:
            raise ValueError("No modality available for inference")

        W = sum(ws)
        return sum(det_probs) / W, sum(type_probs) / W

    def backbone_parameters(self):
        """All parameters except downstream det_heads."""
        det_head_params = set(self.det_heads.parameters())
        return [p for p in self.parameters() if p not in det_head_params]

    def head_parameters(self):
        """Only downstream det_head parameters."""
        return list(self.det_heads.parameters())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:

    def __init__(
        self,
        model: CRLModel,
        loss_fn: SupervisedMultiTaskLoss,
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
            model.parameters(),
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
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        metrics_path = self.save_dir / "crl_metrics.csv"
        fieldnames = [
            "epoch",
            "train_total",
            "train_pres",
            "train_type",
            "train_inst",
            "train_recon",
            "val_total",
            "val_pres",
            "val_type",
            "val_inst",
            "val_recon",
            # Structural quality (computed every 5 epochs)
            "probe_pres_acc",
            "probe_pres_f1",
            "probe_type_acc",
            "probe_type_f1",
            "probe_inst_acc",
            "probe_inst_f1",
            "detection_auc",
        ]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader)
            val_metrics   = self._eval_crl(val_loader)
            self.scheduler.step()

            struct_metrics = {}
            if epoch % 5 == 0:
                struct_metrics = run_full_eval(
                    self.model, train_loader, val_loader, self.device
                )

            row = {
                "epoch":        epoch,
                "train_total":  train_metrics.get("total", 0.0),
                "train_pres":   train_metrics.get("loss_pres", 0.0),
                "train_type":   train_metrics.get("loss_type", 0.0),
                "train_inst":   train_metrics.get("loss_inst", 0.0),
                "train_recon":  train_metrics.get("loss_recon", 0.0),
                "val_total":    val_metrics.get("total", 0.0),
                "val_pres":     val_metrics.get("loss_pres", 0.0),
                "val_type":     val_metrics.get("loss_type", 0.0),
                "val_inst":     val_metrics.get("loss_inst", 0.0),
                "val_recon":    val_metrics.get("loss_recon", 0.0),
                "probe_pres_acc":  struct_metrics.get("probe_pres_acc", ""),
                "probe_pres_f1":   struct_metrics.get("probe_pres_f1", ""),
                "probe_type_acc":  struct_metrics.get("probe_type_acc", ""),
                "probe_type_f1":   struct_metrics.get("probe_type_f1", ""),
                "probe_inst_acc":  struct_metrics.get("probe_inst_acc", ""),
                "probe_inst_f1":   struct_metrics.get("probe_inst_f1", ""),
                "detection_auc":   struct_metrics.get("detection_auc", ""),
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            struct_str = ""
            if struct_metrics:
                struct_str = (
                    f" pres_f1={struct_metrics.get('probe_pres_f1', 0):.3f}"
                    f" type_f1={struct_metrics.get('probe_type_f1', 0):.3f}"
                    f" inst_f1={struct_metrics.get('probe_inst_f1', 0):.3f}"
                    f" auc={struct_metrics.get('detection_auc', 0):.3f}"
                )
            print(
                f"Epoch {epoch:3d} | "
                f"tr={train_metrics.get('total', 0):.4f} "
                f"val={val_metrics.get('total', 0):.4f}"
                f"{struct_str}"
            )

            val_loss = val_metrics.get("total", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_ctr = 0
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / "crl_best.pth",
                )
                print(f"  ✓ New best (val={val_loss:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()
            outputs = self.model.forward(batch, self.device)
            loss, metrics = self.loss_fn(outputs)
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
    def _eval_crl(self, loader: DataLoader) -> dict:
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            outputs = self.model.forward(batch, self.device)
            _, metrics = self.loss_fn(outputs)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        self.model.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Downstream head training
    # ------------------------------------------------------------------

    def train_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        # Reinitialise heads so CRL-phase bias doesn't carry over
        for head in self.model.det_heads.values():
            head.apply(
                lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
            )

        # Freeze CRL backbone (filterbanks, ssms, encoders, decoders, crl_heads)
        for name, p in self.model.named_parameters():
            if "det_heads" not in name:
                p.requires_grad = False

        head_opt = torch.optim.AdamW(self.model.head_parameters(), lr=self.cfg.lr)
        head_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            head_opt, factor=0.5, patience=3
        )

        det_weight  = torch.tensor([1.0, 1.0], device=self.device)
        type_weight = torch.tensor([24.3933, 22.925, 1.42813, 4.64756], device=self.device)
        inst_weight = torch.ones(13, device=self.device)

        metrics_path = self.save_dir / "downstream_metrics.csv"
        fieldnames = [
            "epoch",
            "train_det_loss",
            "train_cls_loss",
            "train_inst_loss",
            "val_det_acc",
            "val_det_f1",
            "val_cls_acc",
            "val_cls_f1",
            "val_inst_acc",
            "val_inst_f1",
        ]
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        best_f1 = 0.0
        patience_ctr = 0

        print("\n=== Downstream Head Training ===")
        for epoch in range(epochs):
            self.model.eval()
            for head in self.model.det_heads.values():
                head.train()

            epoch_det = epoch_cls = epoch_inst = 0.0
            n_det = n_cls = n_inst = 0

            ref = "seismic" if "seismic" in self.model.sensors else self.model.sensors[0]

            for batch in train_loader:
                x_ref  = batch[f"x_{ref}"].to(self.device)
                vtype  = batch["vehicle_type"].to(self.device)
                inst   = batch["instance_type"].to(self.device)
                det    = batch["detection_label"].to(self.device)

                with torch.no_grad():
                    e_pres, e_type, e_inst, _ = self.model.encode_modality(ref, x_ref)

                head_opt.zero_grad()
                pres_logit, type_logits, inst_logits = self.model.det_heads[ref](
                    e_pres, e_type, e_inst
                )

                det_2c = torch.stack([-pres_logit, pres_logit], dim=1)
                det_loss = F.cross_entropy(det_2c, det, weight=det_weight)

                type_mask = vtype >= 0
                cls_loss = torch.tensor(0.0, device=self.device)
                if type_mask.any():
                    cls_loss = F.cross_entropy(
                        type_logits[type_mask], vtype[type_mask], weight=type_weight
                    )

                inst_mask = (det == 1) & (inst >= 0)
                inst_loss = torch.tensor(0.0, device=self.device)
                if inst_mask.any():
                    inst_loss = F.cross_entropy(
                        inst_logits[inst_mask], inst[inst_mask], weight=inst_weight
                    )

                loss = det_loss + cls_loss + inst_loss
                loss.backward()
                head_opt.step()

                epoch_det  += det_loss.item();  n_det  += 1
                epoch_cls  += cls_loss.item() if type_mask.any() else 0.0
                n_cls      += int(type_mask.any())
                epoch_inst += inst_loss.item() if inst_mask.any() else 0.0
                n_inst     += int(inst_mask.any())

            val_m = self._eval_downstream(val_loader)
            head_sched.step(-val_m["val_cls_f1"])

            row = {
                "epoch":          epoch,
                "train_det_loss": epoch_det  / max(n_det,  1),
                "train_cls_loss": epoch_cls  / max(n_cls,  1),
                "train_inst_loss": epoch_inst / max(n_inst, 1),
                **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | "
                f"det={row['train_det_loss']:.4f} "
                f"cls={row['train_cls_loss']:.4f} "
                f"inst={row['train_inst_loss']:.4f} "
                f"det_f1={val_m['val_det_f1']:.4f} "
                f"cls_f1={val_m['val_cls_f1']:.4f} "
                f"inst_f1={val_m['val_inst_f1']:.4f}"
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

        # Post-training diagnostics
        best_head_path = self.save_dir / f"det_head_{ref}_best.pth"
        if best_head_path.exists():
            self.model.det_heads[ref].load_state_dict(
                torch.load(best_head_path, map_location=self.device)
            )
        diag_dir = self.save_dir / "diagnostics"
        print(f"  Generating confusion matrices → {diag_dir}/")
        df_preds = sample_level_eval(self.model, val_loader, self.device, primary_sensor=ref)
        plot_confusion_matrices(df_preds, diag_dir)

    @torch.no_grad()
    def _eval_downstream(self, loader: DataLoader) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()
        det_true, det_pred = [], []
        cls_true, cls_pred = [], []
        inst_true, inst_pred = [], []

        ref = "seismic" if "seismic" in self.model.sensors else self.model.sensors[0]

        for batch in loader:
            x_ref  = batch[f"x_{ref}"].to(self.device)
            vtype  = batch["vehicle_type"]
            inst   = batch["instance_type"]
            det    = batch["detection_label"]

            e_pres, e_type, e_inst, _ = self.model.encode_modality(ref, x_ref)
            pres_logit, type_logits, inst_logits = self.model.det_heads[ref](
                e_pres, e_type, e_inst
            )

            det_pred.extend((pres_logit > 0).long().cpu().tolist())
            det_true.extend(det.tolist())

            type_mask = vtype >= 0
            if type_mask.any():
                cls_pred.extend(
                    type_logits[type_mask.to(self.device)].argmax(1).cpu().tolist()
                )
                cls_true.extend(vtype[type_mask].tolist())

            inst_mask = (det == 1) & (inst >= 0)
            if inst_mask.any():
                inst_pred.extend(
                    inst_logits[inst_mask.to(self.device)].argmax(1).cpu().tolist()
                )
                inst_true.extend(inst[inst_mask].tolist())

        def _f1(true, pred):
            if not true:
                return 0.0, 0.0
            return (
                accuracy_score(true, pred),
                f1_score(true, pred, average="weighted", zero_division=0),
            )

        det_acc,  det_f1  = _f1(det_true,  det_pred)
        cls_acc,  cls_f1  = _f1(cls_true,  cls_pred)
        inst_acc, inst_f1 = _f1(inst_true, inst_pred)

        self.model.train()
        return {
            "val_det_acc":  det_acc,
            "val_det_f1":   det_f1,
            "val_cls_acc":  cls_acc,
            "val_cls_f1":   cls_f1,
            "val_inst_acc": inst_acc,
            "val_inst_f1":  inst_f1,
        }

    @torch.no_grad()
    def calibrate_vote_weights(self, val_loader: DataLoader) -> dict:
        """
        Compute per-modality detection AUC on the validation set and return
        normalised vote weights proportional to (AUC - 0.5).
        """
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        aucs = {}
        for sensor in self.model.sensors:
            scores, labels = [], []
            for batch in val_loader:
                x = batch[f"x_{sensor}"].to(self.device)
                e_pres, e_type, e_inst, _ = self.model.encode_modality(sensor, x)
                pres_logit, _, _ = self.model.det_heads[sensor](e_pres, e_type, e_inst)
                scores.extend(torch.sigmoid(pres_logit).cpu().tolist())
                labels.extend(batch["detection_label"].tolist())
            try:
                auc = roc_auc_score(labels, scores)
            except Exception:
                auc = 0.5
            aucs[sensor] = float(max(0.1, auc - 0.5))

        total = sum(aucs.values())
        weights = {s: v / total for s, v in aucs.items()}
        self.model.train()
        print(f"Vote weights (AUC-based): {weights}")
        return weights
