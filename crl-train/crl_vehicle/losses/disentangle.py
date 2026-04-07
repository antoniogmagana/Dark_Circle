"""
Disentanglement regulariser: penalises off-diagonal correlations in z.

Uses the minibatch off-diagonal correlation penalty (simpler than FactorVAE
TC estimation but effective for moderate batch sizes).
"""

import torch


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
