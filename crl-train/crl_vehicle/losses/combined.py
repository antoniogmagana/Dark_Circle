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
        z_tn_{mod}        : (B, d_z)    latent at t+n (horizon-pair path only)
        z_scm_{mod}       : (B, d_z)    SCM prediction
        avail_{mod}       : (B,) bool   availability mask

    Horizon-pair path (MultiHorizonPairDataset):
        horizon_n         : (B,) long   temporal horizon index n ∈ {1..n_horizons}

    Shared (one SCM per experiment):
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
from crl_vehicle.losses.disentangle import total_correlation_loss, posterior_collapse_penalty


class CombinedLoss(nn.Module):

    def __init__(self, config):
        """config : CRLConfig"""
        super().__init__()
        self.cfg = config
        self.current_beta = config.beta_start
        self.current_lambda_l1 = 0.0
        self.current_lambda_causal = 0.0

    def update_beta(self, epoch: int):
        """Call once per epoch before the training loop."""
        t = min(epoch / max(self.cfg.beta_anneal_epochs, 1), 1.0)
        self.current_beta = (
            self.cfg.beta_start
            + t * (self.cfg.beta_end - self.cfg.beta_start)
        )

    def update_lambda_l1(self, epoch: int):
        """Linearly ramp lambda_l1 from 0 → cfg.lambda_l1_graph over
        lambda_l1_graph_anneal_epochs.  Call once per epoch."""
        t = min(epoch / max(self.cfg.lambda_l1_graph_anneal_epochs, 1), 1.0)
        self.current_lambda_l1 = t * self.cfg.lambda_l1_graph

    def update_lambda_causal(self, epoch: int):
        """Linearly ramp lambda_causal from 0 → cfg.lambda_causal over
        lambda_causal_anneal_epochs.  Call once per epoch."""
        t = min(epoch / max(self.cfg.lambda_causal_anneal_epochs, 1), 1.0)
        self.current_lambda_causal = t * self.cfg.lambda_causal

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
        L_collapse = posterior_collapse_penalty(log_var)

        total = (
            L_recon
            + beta * L_kl
            + self.current_lambda_causal * L_causal
            + self.cfg.lambda_disent * L_disent
            + self.cfg.lambda_collapse * L_collapse
        )
        metrics = {
            f"recon_{mod}": L_recon.item(),
            f"kl_{mod}": L_kl.item(),
            f"causal_{mod}": L_causal.item(),
            f"disent_{mod}": L_disent.item(),
            f"collapse_{mod}": L_collapse.item(),
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

        # --- Graph sparsity (L1 on lower-triangular adjacency weights) ---
        L_l1_graph = outputs.get("scm_l1", torch.tensor(0.0))
        metrics["scm_l1"] = L_l1_graph.item() if torch.is_tensor(L_l1_graph) else float(L_l1_graph)

        _dev = L_l1_graph.device if torch.is_tensor(L_l1_graph) else torch.device("cpu")

        # --- Unknown intervention classifier ---
        L_interv = torch.zeros((), device=_dev)
        interv_logits = outputs.get("interv_logits")
        interv_targets = outputs.get("interv_targets")
        if interv_logits is not None and interv_targets is not None:
            L_interv = F.cross_entropy(interv_logits, interv_targets)
        metrics["interv"] = L_interv.item() if torch.is_tensor(L_interv) else 0.0

        # --- Downstream task losses (summed across modalities) ---
        # Each modality independently predicts presence and type from its own
        # z_presence / z_type block, so both encoders receive direct label gradient.
        vehicle_labels = outputs.get("vehicle_labels")
        det_labels = outputs.get("det_labels")

        L_task = torch.zeros((), device=_dev)
        L_det = torch.zeros((), device=_dev)

        for key, val in outputs.items():
            if key.startswith("vehicle_logits_") and vehicle_labels is not None:
                # Exclude invalid labels (background=-1, multi=-2)
                valid = vehicle_labels >= 0
                if valid.any():
                    L_task = L_task + F.cross_entropy(val[valid], vehicle_labels[valid])
            elif key.startswith("det_logits_") and det_labels is not None:
                L_det = L_det + F.cross_entropy(val, det_labels)

        # Fallback: legacy single-key path (used by downstream Phase 2 trainer)
        if not any(k.startswith("vehicle_logits_") for k in outputs):
            vehicle_logits = outputs.get("vehicle_logits")
            if vehicle_logits is not None and vehicle_labels is not None:
                valid = vehicle_labels >= 0
                if valid.any():
                    L_task = F.cross_entropy(vehicle_logits[valid], vehicle_labels[valid])
        if not any(k.startswith("det_logits_") for k in outputs):
            det_logits = outputs.get("det_logits")
            if det_logits is not None and det_labels is not None:
                L_det = F.cross_entropy(det_logits, det_labels)

        metrics["task_cls"] = L_task.item() if torch.is_tensor(L_task) else 0.0
        metrics["task_det"] = L_det.item() if torch.is_tensor(L_det) else 0.0

        # --- Temporal consistency loss (horizon-pair path only) ---
        # Vehicle-semantic dims (presence, type, proximity) should remain
        # approximately stable across n * horizon_stride_sec seconds.
        # Dividing by n relaxes the constraint for longer horizons.
        L_temporal = torch.zeros((), device=_dev)
        horizon_n = outputs.get("horizon_n")
        for mod in ["audio", "seismic"]:
            z_t_key = f"z_{mod}"
            z_tn_key = f"z_tn_{mod}"
            if z_tn_key not in outputs:
                continue
            avail = outputs.get(f"avail_{mod}")
            z_t = outputs[z_t_key]
            z_tn = outputs[z_tn_key]
            if avail is not None:
                z_t = z_t[avail]
                z_tn = z_tn[avail]
            # Vehicle semantic slice: all dims before noise block
            # noise_start = d_z_presence + d_z_type + d_z_proximity
            noise_start = (
                self.cfg.d_z_presence + self.cfg.d_z_type + self.cfg.d_z_proximity
            )
            z_veh_t  = z_t[:, :noise_start]
            z_veh_tn = z_tn[:, :noise_start]
            sq_err = (z_veh_t - z_veh_tn).pow(2).mean(dim=-1)  # (B',)
            if horizon_n is not None:
                n_float = horizon_n[avail].float() if avail is not None else horizon_n.float()
                sq_err = sq_err / n_float.clamp(min=1.0)
            L_temporal = L_temporal + sq_err.mean()
        metrics["temporal"] = L_temporal.item()

        total = (
            loss_audio
            + loss_seismic
            + self.current_lambda_l1 * L_l1_graph
            + self.cfg.lambda_interv * L_interv
            + self.cfg.lambda_task * (L_task + L_det)
            + self.current_lambda_causal * L_temporal  # reuse causal weight for temporal
        )
        metrics["total"] = total.item()
        metrics["beta"] = beta
        metrics["l1_w"] = self.current_lambda_l1
        metrics["causal_w"] = self.current_lambda_causal
        return total, metrics
