"""
ELBO components: reconstruction loss and KL divergence.
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x_hat: torch.Tensor, x_target: torch.Tensor
) -> torch.Tensor:
    """
    MSE reconstruction loss over filterbank log-envelopes.

    x_hat, x_target: (B, K, T')
    Returns scalar mean loss.
    """
    return F.mse_loss(x_hat, x_target, reduction="mean")


def kl_divergence(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    intervention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I)) per dimension, averaged over batch.

    mu, log_var      : (B, d_z)
    intervention_mask: (B, d_z) binary, 1 = variable was intervened on.
                       Intervened variables bypass KL — their value is
                       externally set, not sampled from the prior.

    Returns scalar.
    """
    # Per-element KL: -0.5 * (1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())   # (B, d_z)

    if intervention_mask is not None:
        kl = kl * (1.0 - intervention_mask.float())

    return kl.mean()
