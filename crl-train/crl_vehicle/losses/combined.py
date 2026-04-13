"""
SupervisedMultiTaskLoss

Three supervised losses applied to the task-specific embeddings produced by
MultiTaskEncoder + CRLHeads during backbone pre-training:

    L_pres  = BCE(pres_logit, detection_label)         — all samples
    L_type  = CE(type_logits, vehicle_type)            — masked: present AND type >= 0
    L_inst  = CE(inst_logits, instance_type)           — masked: present AND inst >= 0
    L_recon = MSE(x_hat, x_target)                    — optional regularizer

    L_total = L_pres + lambda_type * L_type + lambda_inst * L_inst + lambda_recon * L_recon

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


# Type class weights (from training set frequencies — minority classes upweighted)
_TYPE_WEIGHTS = [24.3933, 22.925, 1.42813, 4.64756]

# Instance class weights (uniform — update from training set if frequencies are known)
_INST_WEIGHTS = [1.0] * 13


class SupervisedMultiTaskLoss(nn.Module):

    def __init__(self, config):
        """config : CRLConfig"""
        super().__init__()
        self.cfg = config
        self.register_buffer(
            "type_weight",
            torch.tensor(_TYPE_WEIGHTS, dtype=torch.float32),
        )
        self.register_buffer(
            "inst_weight",
            torch.tensor(_INST_WEIGHTS, dtype=torch.float32),
        )

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

        # Type loss (present vehicles with valid type label only)
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

    def forward(self, outputs: dict) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, metrics_dict).
        metrics_dict contains detached floats for logging.
        """
        det_labels  = outputs["detection_label"]
        type_labels = outputs["vehicle_type"]
        inst_labels = outputs["instance_type"]
        device = det_labels.device

        total_pres  = torch.zeros((), device=device)
        total_type  = torch.zeros((), device=device)
        total_inst  = torch.zeros((), device=device)
        total_recon = torch.zeros((), device=device)
        metrics = {}
        n_sensors = 0

        for mod in ["audio", "seismic"]:
            if f"pres_logit_{mod}" not in outputs:
                continue
            L_pres, L_type, L_inst, L_recon, m = self._modality_losses(
                outputs, mod, det_labels, type_labels, inst_labels, device
            )
            total_pres  = total_pres  + L_pres
            total_type  = total_type  + L_type
            total_inst  = total_inst  + L_inst
            total_recon = total_recon + L_recon
            metrics.update(m)
            n_sensors += 1

        if n_sensors > 1:
            total_pres  = total_pres  / n_sensors
            total_type  = total_type  / n_sensors
            total_inst  = total_inst  / n_sensors
            total_recon = total_recon / n_sensors

        total = (
            total_pres
            + self.cfg.lambda_type  * total_type
            + self.cfg.lambda_inst  * total_inst
            + self.cfg.lambda_recon * total_recon
        )

        metrics["loss_pres"]  = total_pres.item()
        metrics["loss_type"]  = total_type.item()
        metrics["loss_inst"]  = total_inst.item()
        metrics["loss_recon"] = total_recon.item()
        metrics["total"]      = total.item()
        return total, metrics
