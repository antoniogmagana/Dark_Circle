"""
SupervisedMultiTaskLoss

Three supervised losses applied to the task-specific embeddings produced by
MultiTaskEncoder + CRLHeads during backbone pre-training:

    L_pres  = BCE(pres_logit, detection_label)         — all samples
    L_type  = CE(type_logits, vehicle_type)            — masked: present AND type >= 0
    L_inst  = CE(inst_logits, instance_type)           — masked: present AND inst >= 0
    L_recon = MSE(x_hat, x_target)                    — optional regularizer
    L_tc    = off-diagonal correlation penalty on z_veh = [e_pres, e_type]

    L_total = L_pres + lambda_type*L_type + lambda_inst*L_inst
            + lambda_recon*L_recon + lambda_tc*L_tc

Class weights for type and instance losses are computed dynamically from
cumulative sample counts seen during CRL training (not hardcoded).  After
each forward pass the counts are updated and weights are recomputed as
    weight[c] = max_seen_count / count[c]   (0 for classes never seen)
This gives minority classes proportionally higher weight and automatically
zeroes out classes absent from the training split (e.g. iobt-only instances).

Expected keys in the `outputs` dict (built by Trainer.forward):
    Per modality (suffix "_audio" or "_seismic"):
        e_pres_{mod}      : (B, d_pres)  presence embedding
        e_type_{mod}      : (B, d_type)  type embedding
        e_inst_{mod}      : (B, d_inst)  instance embedding
        pres_logit_{mod}  : (B,)         presence logit (from CRLHeads)
        type_logits_{mod} : (B, n_type)  type logits
        inst_logits_{mod} : (B, n_inst)  instance logits
        x_hat_{mod}       : (B, K, T')   reconstructed spectrogram
        x_target_{mod}    : (B, K, T')   stop-grad filterbank target
        avail_{mod}       : (B,) bool    modality availability mask

    Shared:
        detection_label   : (B,) long    0=absent, 1=present
        vehicle_type      : (B,) long    0-3=class, <0=invalid
        instance_type     : (B,) long    0-12=instance, <0=invalid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.losses.disentangle import total_correlation_loss

_N_TYPE = 4
_N_INST = 13


class SupervisedMultiTaskLoss(nn.Module):

    def __init__(self, config):
        """config : CRLConfig"""
        super().__init__()
        self.cfg = config

        # Cumulative sample counts per class — updated each forward() call.
        # Stored as non-grad buffers so they survive checkpoint save/load.
        self.register_buffer("type_counts",  torch.zeros(_N_TYPE, dtype=torch.long))
        self.register_buffer("inst_counts",  torch.zeros(_N_INST, dtype=torch.long))

        # Live class weights derived from counts; start uniform (all-ones).
        # Recomputed in-place after each count update.
        self.register_buffer("type_weight", torch.ones(_N_TYPE,  dtype=torch.float32))
        self.register_buffer("inst_weight", torch.ones(_N_INST,  dtype=torch.float32))

    # ------------------------------------------------------------------
    # Weight maintenance
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        type_labels: torch.Tensor,
        inst_labels: torch.Tensor,
        det_labels: torch.Tensor,
    ) -> None:
        """
        Accumulate per-class sample counts from the current batch, then
        recompute type_weight and inst_weight as max_count / count.

        Only samples that will actually contribute to the respective loss
        are counted:
          - type: present vehicles (det==1) with a valid type label (>=0)
          - inst: present vehicles (det==1) with a valid instance label (>=0)

        Classes never seen retain weight=0, which excludes them from the
        CE normalisation — correct for test-only classes (e.g. polaris/warhog).
        """
        present = det_labels == 1

        # Type counts — valid present samples only
        type_valid = type_labels[present & (type_labels >= 0)]
        if type_valid.numel() > 0:
            self.type_counts += torch.bincount(
                type_valid.cpu(), minlength=_N_TYPE
            ).to(self.type_counts.device)
            max_t = self.type_counts.max().clamp(min=1).float()
            seen_t = self.type_counts > 0
            self.type_weight.zero_()
            self.type_weight[seen_t] = max_t / self.type_counts[seen_t].float()

        # Instance counts — valid present samples only
        inst_valid = inst_labels[present & (inst_labels >= 0)]
        if inst_valid.numel() > 0:
            self.inst_counts += torch.bincount(
                inst_valid.cpu(), minlength=_N_INST
            ).to(self.inst_counts.device)
            max_i = self.inst_counts.max().clamp(min=1).float()
            seen_i = self.inst_counts > 0
            self.inst_weight.zero_()
            self.inst_weight[seen_i] = max_i / self.inst_counts[seen_i].float()

    def _modality_losses(
        self,
        outputs: dict,
        mod: str,
        det_labels: torch.Tensor,
        type_labels: torch.Tensor,
        inst_labels: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compute per-modality losses.
        Returns (L_pres, L_type, L_inst, L_recon, metrics_dict).
        Returns zero tensors if this modality is unavailable in the batch.
        """
        zero = torch.zeros((), device=device)
        avail = outputs.get(f"avail_{mod}")

        if avail is not None and not avail.any():
            return zero, zero, zero, zero, {
                f"pres_{mod}": 0.0,
                f"type_{mod}": 0.0,
                f"inst_{mod}": 0.0,
                f"recon_{mod}": 0.0,
            }

        def sel(key):
            t = outputs[key]
            if avail is not None:
                return t[avail]
            return t

        pres_logit  = sel(f"pres_logit_{mod}")    # (B',)
        type_logits = sel(f"type_logits_{mod}")   # (B', 4)
        inst_logits = sel(f"inst_logits_{mod}")   # (B', 13)
        x_hat       = sel(f"x_hat_{mod}")         # (B', K, T')
        x_target    = sel(f"x_target_{mod}")      # (B', K, T')

        det  = det_labels[avail].float()  if avail is not None else det_labels.float()
        vtype = type_labels[avail]        if avail is not None else type_labels
        inst  = inst_labels[avail]        if avail is not None else inst_labels

        # Presence loss (all samples)
        L_pres = F.binary_cross_entropy_with_logits(pres_logit, det)

        # Type loss (present vehicles with valid type label only).
        # All 4 classes always participate in the softmax so the model learns to
        # separate each type against all others — not just those in the current batch.
        type_mask = (det == 1) & (vtype >= 0)
        if type_mask.any():
            L_type = F.cross_entropy(
                type_logits[type_mask],
                vtype[type_mask],
                weight=self.type_weight,
            )
        else:
            L_type = zero

        # Instance loss (present vehicles with valid instance label only)
        inst_mask = (det == 1) & (inst >= 0)
        if inst_mask.any():
            L_inst = F.cross_entropy(
                inst_logits[inst_mask],
                inst[inst_mask],
                weight=self.inst_weight,
            )
        else:
            L_inst = zero

        # Reconstruction regularizer
        L_recon = F.mse_loss(x_hat, x_target)

        metrics = {
            f"pres_{mod}":  L_pres.item(),
            f"type_{mod}":  L_type.item() if torch.is_tensor(L_type) else 0.0,
            f"inst_{mod}":  L_inst.item() if torch.is_tensor(L_inst) else 0.0,
            f"recon_{mod}": L_recon.item(),
        }
        return L_pres, L_type, L_inst, L_recon, metrics

    def _tc_loss_for_modality(
        self,
        outputs: dict,
        mod: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Total-correlation penalty on z_veh = [e_pres, e_type] for one modality.

        Penalises off-diagonal correlations within and across the two vehicle
        embedding blocks.  This pushes e_type to encode vehicle-class information
        rather than noise characteristics, and reduces z_veh noise leakage.

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

        # Concatenate vehicle embedding blocks and penalise their joint correlations
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
        inst_labels = outputs["instance_type"]
        device = det_labels.device

        if self.training:
            self._update_weights(type_labels, inst_labels, det_labels)

        total_pres  = torch.zeros((), device=device)
        total_type  = torch.zeros((), device=device)
        total_inst  = torch.zeros((), device=device)
        total_recon = torch.zeros((), device=device)
        total_tc    = torch.zeros((), device=device)
        metrics = {}
        n_sensors = 0

        for mod in ["audio", "seismic"]:
            if f"pres_logit_{mod}" not in outputs:
                continue
            L_pres, L_type, L_inst, L_recon, m = self._modality_losses(
                outputs, mod, det_labels, type_labels, inst_labels, device
            )
            L_tc = self._tc_loss_for_modality(outputs, mod, device)
            total_pres  = total_pres  + L_pres
            total_type  = total_type  + L_type
            total_inst  = total_inst  + L_inst
            total_recon = total_recon + L_recon
            total_tc    = total_tc    + L_tc
            metrics.update(m)
            n_sensors += 1

        if n_sensors > 1:
            total_pres  = total_pres  / n_sensors
            total_type  = total_type  / n_sensors
            total_inst  = total_inst  / n_sensors
            total_recon = total_recon / n_sensors
            total_tc    = total_tc    / n_sensors

        total = (
            total_pres
            + self.cfg.lambda_type  * total_type
            + self.cfg.lambda_inst  * total_inst
            + self.cfg.lambda_recon * total_recon
            + self.cfg.lambda_tc    * total_tc
        )

        metrics["loss_pres"]  = total_pres.item()
        metrics["loss_type"]  = total_type.item()
        metrics["loss_inst"]  = total_inst.item()
        metrics["loss_recon"] = total_recon.item()
        metrics["loss_tc"]    = total_tc.item()
        metrics["total"]      = total.item()
        return total, metrics
