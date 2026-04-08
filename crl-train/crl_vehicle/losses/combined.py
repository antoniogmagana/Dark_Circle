"""
CombinedLoss

Aggregates all loss terms from both modalities into a single scalar.
Handles β-annealing of the KL weight.

Expected keys in the `outputs` dict (built by the Trainer per forward pass):
    Per modality (suffix "_audio" or "_seismic"):
        x_hat_{mod}       : (B, K, T')  decoder reconstruction
        x_target_{mod}    : (B, K, T')  filterbank target
        mu_{mod}          : (B, d_z)
        log_var_{mod}     : (B, d_z)
        z_{mod}           : (B, d_z)    reparameterised latent
        z_scm_{mod}       : (B, d_z)    SCM prediction
        avail_{mod}       : (B,) bool   availability mask

    Shared (one SCM per experiment):
        acyclicity        : scalar  — shared SCM acyclicity loss
        interv_mask       : (B, d_z) or None
        interv_logits     : (B, n_targets) or None  — unknown interv cls
        interv_targets    : (B,) long or None

    Downstream:
        vehicle_logits    : (B, n_classes) or None
        vehicle_labels    : (B,) long or None
        det_logits        : (B, 2) or None
        det_labels        : (B,) long or None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.losses.elbo import reconstruction_loss, kl_divergence
from crl_vehicle.losses.causal import scm_consistency_loss
from crl_vehicle.losses.disentangle import total_correlation_loss


class CombinedLoss(nn.Module):

    def __init__(self, config):
        """config : CRLConfig"""
        super().__init__()
        self.cfg = config
        self.current_beta = config.beta_start

    def update_beta(self, epoch: int):
        """Call once per epoch before the training loop."""
        t = min(epoch / max(self.cfg.beta_anneal_epochs, 1), 1.0)
        self.current_beta = (
            self.cfg.beta_start
            + t * (self.cfg.beta_end - self.cfg.beta_start)
        )

    def _modality_terms(
        self, outputs: dict, mod: str, interv_mask: torch.Tensor | None, beta: float
    ) -> tuple[torch.Tensor, dict]:
        """Compute per-modality ELBO + causal + disentanglement losses."""
        avail = outputs.get(f"avail_{mod}")   # (B,) bool or None

        # If the modality is completely absent in this batch, return zeros.
        if avail is not None and not avail.any():
            device = outputs.get("acyclicity", torch.tensor(0.0)).device
            zero = torch.tensor(0.0, device=device)
            return zero, {
                f"recon_{mod}": 0.0,
                f"kl_{mod}": 0.0,
                f"causal_{mod}": 0.0,
                f"disent_{mod}": 0.0,
            }

        # Only compute loss over samples where this modality is available
        def mask_select(key):
            t = outputs[key]
            if avail is not None:
                return t[avail]
            return t

        x_hat = mask_select(f"x_hat_{mod}")
        x_tgt = mask_select(f"x_target_{mod}")
        mu = mask_select(f"mu_{mod}")
        log_var = mask_select(f"log_var_{mod}")
        z = mask_select(f"z_{mod}")
        z_scm = mask_select(f"z_scm_{mod}")
        im = mask_select("interv_mask") if interv_mask is not None else None

        L_recon = reconstruction_loss(x_hat, x_tgt)
        L_kl = kl_divergence(mu, log_var, im)
        L_causal = scm_consistency_loss(z, z_scm)
        L_disent = total_correlation_loss(z)

        total = (
            L_recon
            + beta * L_kl
            + self.cfg.lambda_causal * L_causal
            + self.cfg.lambda_disent * L_disent
        )
        metrics = {
            f"recon_{mod}": L_recon.item(),
            f"kl_{mod}": L_kl.item(),
            f"causal_{mod}": L_causal.item(),
            f"disent_{mod}": L_disent.item(),
        }
        return total, metrics

    def forward(
        self, outputs: dict, beta_override: float | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, metrics_dict).
        metrics_dict contains detached floats for logging.

        Args:
            beta_override: If provided, use this beta instead of current_beta.
                           Pass cfg.beta_end to compute a fixed-beta loss that
                           is comparable across epochs regardless of annealing
                           schedule (used for checkpointing).
        """
        beta = beta_override if beta_override is not None else self.current_beta
        interv_mask = outputs.get("interv_mask")
        metrics = {}

        # --- Per-modality terms ---
        loss_audio, m_audio = self._modality_terms(outputs, "audio", interv_mask, beta)
        loss_seismic, m_seismic = self._modality_terms(outputs, "seismic", interv_mask, beta)
        metrics.update(m_audio)
        metrics.update(m_seismic)

        # --- Acyclicity (shared SCM) ---
        L_acyclic = outputs.get("acyclicity", torch.tensor(0.0))
        metrics["acyclic"] = L_acyclic.item() if torch.is_tensor(L_acyclic) else float(L_acyclic)

        _dev = L_acyclic.device if torch.is_tensor(L_acyclic) else torch.device("cpu")

        # --- Unknown intervention classifier ---
        L_interv = torch.zeros((), device=_dev)
        interv_logits = outputs.get("interv_logits")
        interv_targets = outputs.get("interv_targets")
        if interv_logits is not None and interv_targets is not None:
            L_interv = F.cross_entropy(interv_logits, interv_targets)
        metrics["interv"] = L_interv.item() if torch.is_tensor(L_interv) else 0.0

        # --- Downstream task losses ---
        L_task = torch.zeros((), device=_dev)
        vehicle_logits = outputs.get("vehicle_logits")
        vehicle_labels = outputs.get("vehicle_labels")
        if vehicle_logits is not None and vehicle_labels is not None:
            # Exclude invalid labels (background=-1, multi=-2)
            valid = vehicle_labels >= 0
            if valid.any():
                L_task = F.cross_entropy(
                    vehicle_logits[valid], vehicle_labels[valid]
                )
        metrics["task_cls"] = L_task.item() if torch.is_tensor(L_task) else 0.0

        L_det = torch.zeros((), device=_dev)
        det_logits = outputs.get("det_logits")
        det_labels = outputs.get("det_labels")
        if det_logits is not None and det_labels is not None:
            L_det = F.cross_entropy(det_logits, det_labels)
        metrics["task_det"] = L_det.item() if torch.is_tensor(L_det) else 0.0

        total = (
            loss_audio
            + loss_seismic
            + self.cfg.lambda_acyclic * L_acyclic
            + self.cfg.lambda_interv * L_interv
            + self.cfg.lambda_task * (L_task + L_det)
        )
        metrics["total"] = total.item()
        metrics["beta"] = beta
        return total, metrics
