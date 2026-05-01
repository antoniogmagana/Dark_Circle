from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss (scalar)."""
    return F.mse_loss(x_hat, x)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy: (1 - p_t)^gamma * weighted CE, mean-reduced.

    Stacks the focal modulator on top of class weights. With gamma=0 this
    reduces exactly to F.cross_entropy(weight=weight). With gamma>0,
    confident-correct samples (p_t near 1) contribute less; uncertain ones
    (p_t near 1/n_classes) contribute more.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal = (1.0 - pt).pow(gamma)
    nll = -log_pt
    if weight is not None:
        w = weight[target]
        return (focal * w * nll).sum() / w.sum().clamp_min(1e-12)
    return (focal * nll).mean()


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """KL[q(z|x) || N(0,I)], summed over latent dims, meaned over batch, scaled by beta."""
    kl = 0.5 * (log_var.exp() + mu**2 - 1 - log_var).sum(dim=-1)
    return beta * kl.mean()


def intervention_matching_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE intervention matching loss.

    logits:  (B, 2) — raw logits for [pres_changed, type_changed]
    targets: (B, 2) float — binary ground truth from label_change_target()
    Returns scalar (mean over batch and bits).
    """
    return F.binary_cross_entropy_with_logits(logits, targets)
