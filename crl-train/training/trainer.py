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
                    self.model, train_loader, val_loader, self.device,
                    max_batches=self.cfg.steps_per_epoch,
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
            if self.cfg.steps_per_epoch and n >= self.cfg.steps_per_epoch:
                break
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

        for sensor in self.model.sensors:
            self._train_sensor_downstream(sensor, train_loader, val_loader, epochs)

        self._fused_eval(val_loader)

    def _train_sensor_downstream(
        self,
        sensor: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        det_head = self.model.det_heads[sensor]

        # Independent optimizer + scheduler per task head
        opt_det  = torch.optim.AdamW(det_head.presence.parameters(),      lr=self.cfg.lr)
        opt_cls  = torch.optim.AdamW(det_head.type_head.parameters(),     lr=self.cfg.lr)
        opt_inst = torch.optim.AdamW(det_head.instance_head.parameters(), lr=self.cfg.lr)

        sched_det  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_det,  factor=0.5, patience=3)
        sched_cls  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_cls,  factor=0.5, patience=3)
        sched_inst = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_inst, factor=0.5, patience=3)

        det_weight  = torch.tensor([1.0, 1.0], device=self.device)
        type_weight = torch.tensor([24.3933, 22.925, 1.42813, 4.64756], device=self.device)
        inst_weight = torch.ones(13, device=self.device)

        metrics_path = self.save_dir / f"downstream_metrics_{sensor}.csv"
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

        # Independent best/patience tracking per task head
        best_det_f1  = best_cls_f1 = best_inst_f1 = 0.0
        pat_det = pat_cls = pat_inst = 0
        stopped_det = stopped_cls = stopped_inst = False

        print(f"\n=== Downstream Head Training [{sensor}] ===")
        for epoch in range(epochs):
            if stopped_det and stopped_cls and stopped_inst:
                break

            self.model.eval()
            det_head.train()

            epoch_det = epoch_cls = epoch_inst = 0.0
            n_det = n_cls = n_inst = 0

            n_train_steps = 0
            for batch in train_loader:
                avail = batch[f"{sensor}_avail"]
                if not avail.any():
                    n_train_steps += 1
                    if self.cfg.steps_per_epoch and n_train_steps >= self.cfg.steps_per_epoch:
                        break
                    continue

                x_sensor = batch[f"x_{sensor}"][avail].to(self.device)
                vtype = batch["vehicle_type"][avail].to(self.device)
                inst  = batch["instance_type"][avail].to(self.device)
                det   = batch["detection_label"][avail].to(self.device)

                with torch.no_grad():
                    e_pres, e_type, e_inst, _ = self.model.encode_modality(sensor, x_sensor)

                # Detection head — all available samples
                if not stopped_det:
                    opt_det.zero_grad()
                    pres_logit = det_head.presence(e_pres)
                    det_2c = torch.stack([-pres_logit, pres_logit], dim=1)
                    det_loss = F.cross_entropy(det_2c, det, weight=det_weight)
                    det_loss.backward()
                    opt_det.step()
                    epoch_det += det_loss.item()
                    n_det += 1
                else:
                    with torch.no_grad():
                        pres_logit = det_head.presence(e_pres)

                # Type head — detected-present samples with valid type label
                type_mask = (det == 1) & (vtype >= 0)
                if not stopped_cls and type_mask.any():
                    opt_cls.zero_grad()
                    type_logits = det_head.type_head(e_type)
                    cls_loss = F.cross_entropy(
                        type_logits[type_mask], vtype[type_mask], weight=type_weight
                    )
                    cls_loss.backward()
                    opt_cls.step()
                    epoch_cls += cls_loss.item()
                    n_cls += 1

                # Instance head — detected-present samples with valid instance label
                inst_mask = (det == 1) & (inst >= 0)
                if not stopped_inst and inst_mask.any():
                    opt_inst.zero_grad()
                    inst_logits = det_head.instance_head(e_inst)
                    inst_loss = F.cross_entropy(
                        inst_logits[inst_mask], inst[inst_mask], weight=inst_weight
                    )
                    inst_loss.backward()
                    opt_inst.step()
                    epoch_inst += inst_loss.item()
                    n_inst += 1

                n_train_steps += 1
                if self.cfg.steps_per_epoch and n_train_steps >= self.cfg.steps_per_epoch:
                    break

            val_m = self._eval_downstream(val_loader, sensor)
            sched_det.step(-val_m["val_det_f1"])
            sched_cls.step(-val_m["val_cls_f1"])
            sched_inst.step(-val_m["val_inst_f1"])

            row = {
                "epoch":           epoch,
                "train_det_loss":  epoch_det  / max(n_det,  1),
                "train_cls_loss":  epoch_cls  / max(n_cls,  1),
                "train_inst_loss": epoch_inst / max(n_inst, 1),
                **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            status = []

            if val_m["val_det_f1"] > best_det_f1:
                best_det_f1 = val_m["val_det_f1"]
                pat_det = 0
                torch.save(
                    det_head.presence.state_dict(),
                    self.save_dir / f"presence_head_{sensor}_best.pth",
                )
                status.append(f"det✓ F1={best_det_f1:.4f}")
            elif not stopped_det:
                pat_det += 1
                if pat_det >= self.cfg.early_stop_patience:
                    stopped_det = True
                    status.append(f"det stopped (ep {epoch})")

            if val_m["val_cls_f1"] > best_cls_f1:
                best_cls_f1 = val_m["val_cls_f1"]
                pat_cls = 0
                torch.save(
                    det_head.type_head.state_dict(),
                    self.save_dir / f"type_head_{sensor}_best.pth",
                )
                status.append(f"cls✓ F1={best_cls_f1:.4f}")
            elif not stopped_cls:
                pat_cls += 1
                if pat_cls >= self.cfg.early_stop_patience:
                    stopped_cls = True
                    status.append(f"cls stopped (ep {epoch})")

            if val_m["val_inst_f1"] > best_inst_f1:
                best_inst_f1 = val_m["val_inst_f1"]
                pat_inst = 0
                torch.save(
                    det_head.instance_head.state_dict(),
                    self.save_dir / f"inst_head_{sensor}_best.pth",
                )
                status.append(f"inst✓ F1={best_inst_f1:.4f}")
            elif not stopped_inst:
                pat_inst += 1
                if pat_inst >= self.cfg.early_stop_patience:
                    stopped_inst = True
                    status.append(f"inst stopped (ep {epoch})")

            print(
                f"Epoch {epoch:3d} | "
                f"det={row['train_det_loss']:.4f} "
                f"cls={row['train_cls_loss']:.4f} "
                f"inst={row['train_inst_loss']:.4f} | "
                f"det_f1={val_m['val_det_f1']:.4f} "
                f"cls_f1={val_m['val_cls_f1']:.4f} "
                f"inst_f1={val_m['val_inst_f1']:.4f}"
                + (f"  [{', '.join(status)}]" if status else "")
            )

        print(
            f"[{sensor}] Downstream complete. "
            f"Best det F1={best_det_f1:.4f}  "
            f"cls F1={best_cls_f1:.4f}  "
            f"inst F1={best_inst_f1:.4f}"
        )

        # Load best weights for diagnostics
        for fname, subhead in [
            (f"presence_head_{sensor}_best.pth", det_head.presence),
            (f"type_head_{sensor}_best.pth",     det_head.type_head),
            (f"inst_head_{sensor}_best.pth",     det_head.instance_head),
        ]:
            p = self.save_dir / fname
            if p.exists():
                subhead.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))

        diag_dir = self.save_dir / f"diagnostics_{sensor}"
        print(f"  Generating confusion matrices → {diag_dir}/")
        df_preds = sample_level_eval(
            self.model, val_loader, self.device,
            primary_sensor=sensor,
            max_batches=self.cfg.steps_per_epoch,
        )
        plot_confusion_matrices(df_preds, diag_dir)

    @torch.no_grad()
    def _eval_downstream(self, loader: DataLoader, sensor: str) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()
        det_true, det_pred = [], []
        cls_true, cls_pred = [], []
        inst_true, inst_pred = [], []

        for n_eval, batch in enumerate(loader):
            if self.cfg.steps_per_epoch and n_eval >= self.cfg.steps_per_epoch:
                break
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue

            x_sensor = batch[f"x_{sensor}"][avail].to(self.device)
            vtype = batch["vehicle_type"][avail]
            inst  = batch["instance_type"][avail]
            det   = batch["detection_label"][avail]

            e_pres, e_type, e_inst, _ = self.model.encode_modality(sensor, x_sensor)
            pres_logit, type_logits, inst_logits = self.model.det_heads[sensor](
                e_pres, e_type, e_inst
            )

            det_pred.extend((pres_logit > 0).long().cpu().tolist())
            det_true.extend(det.tolist())

            type_mask = (det == 1) & (vtype >= 0)
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
    def _fused_eval(self, val_loader: DataLoader) -> None:
        """
        Final verification pass after all per-sensor downstream heads are trained.

        Averages predictions across available sensors (weighted equally) and
        computes overall F1 per task head (presence, type, instance).  Saves
        confusion matrices and a per-sample CSV to diagnostics_fused/.

        Detection:  average sigmoid(pres_logit) across sensors, threshold at 0.5.
        Type/inst:  average softmax(logits) across sensors, argmax.
                    Only evaluated on ground-truth det==1 samples with valid labels.
        """
        from sklearn.metrics import f1_score, accuracy_score
        from training.eval import plot_confusion_matrices

        import pandas as pd

        self.model.eval()

        det_true, det_pred_fused = [], []
        type_true, type_pred_fused = [], []
        inst_true, inst_pred_fused = [], []
        rows = []

        n_eval = 0
        for batch in val_loader:
            if self.cfg.steps_per_epoch and n_eval >= self.cfg.steps_per_epoch:
                break

            det_gt   = batch["detection_label"]
            vtype_gt = batch["vehicle_type"]
            inst_gt  = batch["instance_type"]
            seg_ids  = batch["segment_id"].tolist()
            intervs  = batch["interv_idx"].tolist()
            B        = det_gt.shape[0]

            # Accumulate per-sensor probability tensors
            ref_head   = self.model.det_heads[self.model.sensors[0]]
            n_types    = ref_head.type_head.head[-1].out_features
            n_insts    = ref_head.instance_head.head[-1].out_features
            det_probs  = torch.zeros(B)
            type_probs = torch.zeros(B, n_types)
            inst_probs = torch.zeros(B, n_insts)
            n_sensors  = torch.zeros(B)

            for sensor in self.model.sensors:
                avail = batch[f"{sensor}_avail"]
                if not avail.any():
                    continue
                x_s = batch[f"x_{sensor}"][avail].to(self.device)
                e_pres, e_type, e_inst, _ = self.model.encode_modality(sensor, x_s)
                pres_logit, type_logits, inst_logits = self.model.det_heads[sensor](
                    e_pres, e_type, e_inst
                )
                det_probs[avail]  += torch.sigmoid(pres_logit).cpu()
                type_probs[avail] += torch.softmax(type_logits, dim=-1).cpu()
                inst_probs[avail] += torch.softmax(inst_logits, dim=-1).cpu()
                n_sensors[avail]  += 1

            # Samples with at least one available sensor
            any_avail = n_sensors > 0
            if not any_avail.any():
                n_eval += 1
                continue

            # Normalise by number of contributing sensors
            n_s = n_sensors[any_avail].unsqueeze(1)
            det_p_norm   = (det_probs[any_avail] / n_sensors[any_avail]).numpy()
            type_p_norm  = (type_probs[any_avail] / n_s).numpy()
            inst_p_norm  = (inst_probs[any_avail] / n_s).numpy()

            det_pred  = (det_p_norm >= 0.5).astype(int)
            type_pred = type_p_norm.argmax(axis=1)
            inst_pred = inst_p_norm.argmax(axis=1)

            det_gt_sub   = det_gt[any_avail].numpy()
            vtype_gt_sub = vtype_gt[any_avail].numpy()
            inst_gt_sub  = inst_gt[any_avail].numpy()

            det_true.extend(det_gt_sub.tolist())
            det_pred_fused.extend(det_pred.tolist())

            type_mask = (det_gt_sub == 1) & (vtype_gt_sub >= 0)
            type_true.extend(vtype_gt_sub[type_mask].tolist())
            type_pred_fused.extend(type_pred[type_mask].tolist())

            inst_mask = (det_gt_sub == 1) & (inst_gt_sub >= 0)
            inst_true.extend(inst_gt_sub[inst_mask].tolist())
            inst_pred_fused.extend(inst_pred[inst_mask].tolist())

            seg_ids_sub  = [seg_ids[i]  for i in range(B) if any_avail[i]]
            intervs_sub  = [intervs[i]  for i in range(B) if any_avail[i]]
            for j in range(len(seg_ids_sub)):
                ti = int(vtype_gt_sub[j])
                ii = int(inst_gt_sub[j])
                rows.append({
                    "segment_id": seg_ids_sub[j],
                    "interv_idx": intervs_sub[j],
                    "true_det":   int(det_gt_sub[j]),
                    "pred_det":   int(det_pred[j]),
                    "true_type":  ti,
                    "pred_type":  int(type_pred[j]) if ti >= 0 else -1,
                    "true_inst":  ii,
                    "pred_inst":  int(inst_pred[j]) if ii >= 0 else -1,
                })

            n_eval += 1

        self.model.train()

        def _metrics(true, pred, name):
            if not true:
                print(f"  [fused] {name}: no valid samples")
                return 0.0, 0.0
            acc = accuracy_score(true, pred)
            f1  = f1_score(true, pred, average="weighted", zero_division=0)
            print(f"  [fused] {name}: acc={acc:.4f}  f1={f1:.4f}")
            return acc, f1

        print("\n=== Fused Evaluation (all sensors combined) ===")
        _metrics(det_true,  det_pred_fused,  "presence")
        _metrics(type_true, type_pred_fused, "type")
        _metrics(inst_true, inst_pred_fused, "instance")

        diag_dir = self.save_dir / "diagnostics_fused"
        diag_dir.mkdir(parents=True, exist_ok=True)
        df_preds = pd.DataFrame(rows)
        if not df_preds.empty:
            plot_confusion_matrices(df_preds, diag_dir, tag="fused")
            print(f"  Confusion matrices → {diag_dir}/")

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
