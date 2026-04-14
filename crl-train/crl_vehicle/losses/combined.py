"""
SupervisedMultiTaskLoss

Two supervised losses applied to the task-specific embeddings produced by
MultiTaskEncoder + CRLHeads during backbone pre-training:

    L_pres = BCE(pres_logit, detection_label)         — all samples
    L_type = CE(type_logits, vehicle_type)            — masked: present AND type >= 0
    L_tc   = off-diagonal correlation penalty on z_veh = [e_pres, e_type]

    L_total = L_pres + lambda_type*L_type + lambda_tc*L_tc

Class weights for the type loss are computed dynamically from cumulative
sample counts seen during CRL training (not hardcoded).  After each forward
pass the counts are updated and weights are recomputed as
    weight[c] = max_seen_count / count[c]   (0 for classes never seen)
This gives minority classes proportionally higher weight and automatically
zeroes out classes absent from the training split.

Expected keys in the `outputs` dict (built by Trainer.forward):
    Per modality (suffix "_audio" or "_seismic"):
        e_pres_{mod}      : (B, d_pres)  presence embedding
        e_type_{mod}      : (B, d_type)  type embedding
        pres_logit_{mod}  : (B,)         presence logit (from CRLHeads)
        type_logits_{mod} : (B, n_type)  type logits
        avail_{mod}       : (B,) bool    modality availability mask

    Shared:
        detection_label   : (B,) long    0=absent, 1=present
        vehicle_type      : (B,) long    0-3=class, <0=invalid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.losses.disentangle import total_correlation_loss

_N_TYPE = 4


class SupervisedMultiTaskLoss(nn.Module):

    def __init__(self, config):
        """
        config : CRLConfig
        """
        super().__init__()
        self.cfg = config

        # Cumulative sample counts per class — updated each forward() call.
        # Stored as non-grad buffers so they survive checkpoint save/load.
        self.register_buffer("type_counts", torch.zeros(_N_TYPE, dtype=torch.long))

        # Live class weights derived from counts; start uniform (all-ones).
        # Recomputed in-place after each count update.
        self.register_buffer("type_weight", torch.ones(_N_TYPE, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Weight maintenance
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        type_labels: torch.Tensor,
        det_labels: torch.Tensor,
    ) -> None:
        """
        Accumulate per-class sample counts from the current batch, then
        recompute type_weight as max_count / count.

        Only samples that will actually contribute to the loss are counted:
          - type: present vehicles (det==1) with a valid type label (>=0)

        Classes never seen retain weight=0, which excludes them from the
        CE normalisation — correct for test-only classes.
        """
        present = det_labels == 1

        type_valid = type_labels[present & (type_labels >= 0)]
        if type_valid.numel() > 0:
            self.type_counts += torch.bincount(
                type_valid.cpu(), minlength=_N_TYPE
            ).to(self.type_counts.device)
            max_t = self.type_counts.max().clamp(min=1).float()
            seen_t = self.type_counts > 0
            self.type_weight.zero_()
            self.type_weight[seen_t] = max_t / self.type_counts[seen_t].float()

    def _modality_losses(
        self,
        outputs: dict,
        mod: str,
        det_labels: torch.Tensor,
        type_labels: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute per-modality losses.
        Returns (L_pres, L_type, metrics_dict).
        Returns zero tensors if this modality is unavailable in the batch.
        """
        zero = torch.zeros((), device=device)
        avail = outputs.get(f"avail_{mod}")

        if avail is not None and not avail.any():
            return zero, zero, {
                f"pres_{mod}": 0.0,
                f"type_{mod}": 0.0,
            }

        def sel(key):
            t = outputs[key]
            if avail is not None:
                return t[avail]
            return t

        pres_logit  = sel(f"pres_logit_{mod}")    # (B',)
        type_logits = sel(f"type_logits_{mod}")   # (B', 4)

        det   = det_labels[avail].float()  if avail is not None else det_labels.float()
        vtype = type_labels[avail]         if avail is not None else type_labels

        # Presence loss (all samples)
        L_pres = F.binary_cross_entropy_with_logits(pres_logit, det)

        # Type loss (present vehicles with valid type label only).
        type_mask = (det == 1) & (vtype >= 0)
        if type_mask.any():
            L_type = F.cross_entropy(
                type_logits[type_mask],
                vtype[type_mask],
                weight=self.type_weight,
            )
        else:
            L_type = zero

        metrics = {
            f"pres_{mod}": L_pres.item(),
            f"type_{mod}": L_type.item() if torch.is_tensor(L_type) else 0.0,
        }
        return L_pres, L_type, metrics

    def _tc_loss_for_modality(
        self,
        outputs: dict,
        mod: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Total-correlation penalty on z_veh = [e_pres, e_type] for one modality.
        Applied only when ≥ 2 samples from this modality are available.
        Returns a scalar tensor (0 if modality unavailable).
        """
        zero = torch.zeros((), device=device)
        avail = outputs.get(f"avail_{mod}")

        e_pres = outputs.get(f"e_pres_{mod}")
        e_type = outputs.get(f"e_type_{mod}")
        if e_pres is None or e_type is None:
            return zero

        if avail is not None:
            e_pres = e_pres[avail]
            e_type = e_type[avail]

        if e_pres.shape[0] < 2:
            return zero

        z_veh = torch.cat([e_pres, e_type], dim=-1)  # (B', d_pres + d_type)
        return total_correlation_loss(z_veh)

    def forward(self, outputs: dict) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, metrics_dict).
        metrics_dict contains detached floats for logging.

        When the model is in training mode, cumulative class counts are updated
        from this batch before the weights are applied to the loss.  In eval
        mode the counts and weights are frozen so val-set class frequencies do
        not contaminate the training-derived weights.
        """
        det_labels  = outputs["detection_label"]
        type_labels = outputs["vehicle_type"]
        device = det_labels.device

        if self.training:
            self._update_weights(type_labels, det_labels)

        total_pres = torch.zeros((), device=device)
        total_type = torch.zeros((), device=device)
        total_tc   = torch.zeros((), device=device)
        metrics = {}
        n_sensors = 0

        for mod in ["audio", "seismic"]:
            if f"pres_logit_{mod}" not in outputs:
                continue
            L_pres, L_type, m = self._modality_losses(
                outputs, mod, det_labels, type_labels, device
            )
            L_tc = self._tc_loss_for_modality(outputs, mod, device)
            total_pres = total_pres + L_pres
            total_type = total_type + L_type
            total_tc   = total_tc   + L_tc
            metrics.update(m)
            n_sensors += 1

        if n_sensors > 1:
            total_pres = total_pres / n_sensors
            total_type = total_type / n_sensors
            total_tc   = total_tc   / n_sensors

        total = (
            total_pres
            + self.cfg.lambda_type * total_type
            + self.cfg.lambda_tc   * total_tc
        )

        metrics["loss_pres"] = total_pres.item()
        metrics["loss_type"] = total_type.item()
        metrics["loss_tc"]   = total_tc.item()
        metrics["total"]     = total.item()
        return total, metrics
