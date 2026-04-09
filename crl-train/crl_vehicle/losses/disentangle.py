"""
Disentanglement regulariser: penalises off-diagonal correlations in z.

Uses the minibatch off-diagonal correlation penalty (simpler than FactorVAE
TC estimation but effective for moderate batch sizes).
"""

import torch
import torch.nn.functional as F


def total_correlation_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Off-diagonal correlation penalty — approximates total correlation.

    Normalise z columns to zero mean, unit variance, then compute the
    (d_z, d_z) correlation matrix.  The loss is the mean squared value of
    all off-diagonal entries.

    z : (B, d_z)
    Returns scalar.  = 0 when all z dimensions are uncorrelated.
    """
    B, d_z = z.shape
    z_norm = (z - z.mean(0)) / (z.std(0) + 1e-8)   # (B, d_z)
    cov = (z_norm.T @ z_norm) / B                    # (d_z, d_z)
    off_diag = cov - torch.diag(cov.diag())
    return off_diag.pow(2).mean()


def posterior_collapse_penalty(
    log_var: torch.Tensor, threshold: float = -3.0
) -> torch.Tensor:
    """
    Penalises latent dimensions where log_var < threshold.

    In beta-VAE training, unused dimensions collapse to a deterministic
    unit (log_var → -∞).  This penalty forces all d_z dimensions to remain
    stochastic, ensuring neither vehicle-signature nor environmental dimensions
    become trivially constant.

    log_var  : (B, d_z)
    threshold: log-variance below this value is penalised (default -3 → σ≈0.22)
    Returns scalar ≥ 0.
    """
    return F.relu(threshold - log_var).mean()
