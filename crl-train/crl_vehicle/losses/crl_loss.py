from __future__ import annotations
import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss (scalar)."""
    return F.mse_loss(x_hat, x)


def kl_divergence(
    mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """KL[q(z|x) || N(0,I)], summed over latent dims, meaned over batch, scaled by beta."""
    kl = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(dim=-1)
    return beta * kl.mean()


def intervention_matching_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """BCE intervention matching loss.

    logits:  (B, 2) — raw logits for [pres_changed, type_changed]
    targets: (B, 2) float — binary ground truth from label_change_target()
    Returns scalar (mean over batch and bits).
    """
    return F.binary_cross_entropy_with_logits(logits, targets)
