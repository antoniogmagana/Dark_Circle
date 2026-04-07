"""
Causal consistency loss: encoder output should agree with SCM prediction.
"""

import torch
import torch.nn.functional as F


def scm_consistency_loss(
    z: torch.Tensor, z_hat_scm: torch.Tensor
) -> torch.Tensor:
    """
    Penalises the difference between the encoder-sampled z and the SCM's
    causal prediction z_hat for non-intervened variables.

    The SCM forward already zeros out the mechanism output for intervened
    variables (replacing them with z), so this MSE directly measures how
    well the causal graph predicts the latent structure.

    z, z_hat_scm : (B, d_z)
    Returns scalar.
    """
    return F.mse_loss(z_hat_scm, z, reduction="mean")
