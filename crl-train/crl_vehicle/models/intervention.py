from __future__ import annotations
import torch
import torch.nn as nn
from crl_vehicle.models.latent import CausalLatentSpace


def label_change_target(
    det_t: torch.Tensor,
    det_tn: torch.Tensor,
    type_t: torch.Tensor,
    type_tn: torch.Tensor,
) -> torch.Tensor:
    """Compute (B, 2) binary float targets: [pres_changed, type_changed]."""
    pres_changed = (det_t != det_tn).float()
    valid = (type_t >= 0) | (type_tn >= 0)
    type_changed = (valid & (type_t != type_tn)).float()
    return torch.stack([pres_changed, type_changed], dim=1)


class UnknownInterventionClassifier(nn.Module):
    """MLP predicting which latent block changed between t and t+1.
    Operates on the ENV block (D_ENV=6 dims)."""

    def __init__(
        self, d_env: int = CausalLatentSpace.D_ENV, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_env, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, z_env_t: torch.Tensor, z_env_tn: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(torch.cat([z_env_t, z_env_tn], dim=-1))
