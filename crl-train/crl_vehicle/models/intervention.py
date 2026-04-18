"""
UnknownInterventionClassifier (redesigned)

Given z_env slices from two consecutive timesteps, predicts a 2-bit
label-change vector: [pres_changed, type_changed].

This is structurally valid CITRIS pressure: the classifier learns which
causal variable changed between t and t+1, pushing presence signal into
z_pres and type signal into z_type.

Noise augmentations are retained as data augmentation but are NOT part of
the intervention target — they are decoupled from causal structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_ENV = 6   # must match CausalLatentSpace.D_ENV


def label_change_target(
    det_t: torch.Tensor,
    det_tn: torch.Tensor,
    type_t: torch.Tensor,
    type_tn: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 2-bit change vector from consecutive ground-truth labels.

    det_t, det_tn  : (B,) long — detection_label {0, 1}
    type_t, type_tn: (B,) long — vehicle_type {-2, -1, 0..3}

    Returns: (B, 2) float tensor
        col 0 = pres_changed  (detection_label differs)
        col 1 = type_changed  (vehicle_type differs, ignoring background/multi)
    """
    pres_changed = (det_t != det_tn).float()

    # Only meaningful when at least one window has a valid vehicle type
    valid = (type_t >= 0) | (type_tn >= 0)
    type_changed = ((type_t != type_tn) & valid).float()

    return torch.stack([pres_changed, type_changed], dim=-1)   # (B, 2)


class UnknownInterventionClassifier(nn.Module):
    """
    MLP: given (z_env_t, z_env_tn), predict [pres_changed, type_changed].

    Input : cat([z_env_t, z_env_tn])  →  (B, 2 * D_ENV) = (B, 12)
    Output: 2 logits                  →  (B, 2)
    Loss  : BCE per bit (not softmax)
    """

    def __init__(self, d_env: int = D_ENV, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_env, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_env_t: torch.Tensor, z_env_tn: torch.Tensor) -> torch.Tensor:
        """
        z_env_t, z_env_tn : (B, D_ENV)
        Returns            : (B, 2) logits
        """
        x = torch.cat([z_env_t, z_env_tn], dim=-1)
        return self.classifier(x)
