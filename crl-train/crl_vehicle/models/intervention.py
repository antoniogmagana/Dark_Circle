"""
UnknownInterventionClassifier

Given two consecutive latent vectors (z_t, z_tn), predicts which causal block
changed between the two time steps.

Target classes (5 total):
    0 = no change          — neither window was intervened on
    1 = presence changed
    2 = type changed
    3 = proximity changed
    4 = noise changed      — any of the 7 synthetic noise interventions (interv_idx 1-7)

All 7 intervention types in transforms.py are environmental noise injections,
so they all map to class 4 (noise block). Presence, type, and proximity are
never directly intervened on by the data pipeline — those targets are reserved
for future use and will not appear during training with the current dataset.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Target mapping
# ---------------------------------------------------------------------------

# Block target indices
BLOCK_NO_CHANGE  = 0
BLOCK_PRESENCE   = 1
BLOCK_TYPE       = 2
BLOCK_PROXIMITY  = 3
BLOCK_NOISE      = 4

N_BLOCK_TARGETS  = 5


def interv_idx_to_block_target(interv_idx_t: torch.Tensor, interv_idx_tn: torch.Tensor) -> torch.Tensor:
    """
    Map per-sample intervention indices at t and t+n to a block-level target.

    Rules:
        - If neither step was intervened on → BLOCK_NO_CHANGE (0)
        - If either step was intervened on (interv_idx > 0) → BLOCK_NOISE (4)
          (all 7 synthetic interventions are noise injections)

    interv_idx_t, interv_idx_tn : (B,) long tensors, values in {0..7}
    Returns                      : (B,) long tensor of block targets
    """
    either_intervened = (interv_idx_t > 0) | (interv_idx_tn > 0)
    targets = torch.where(either_intervened,
                          torch.full_like(interv_idx_t, BLOCK_NOISE),
                          torch.full_like(interv_idx_t, BLOCK_NO_CHANGE))
    return targets


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class UnknownInterventionClassifier(nn.Module):
    """
    CITRIS-style MLP: given (z_t, z_tn), predict which latent block changed.

    Input : concatenation [z_t, z_tn]  →  (B, 2 * d_z)
    Output: logits over N_BLOCK_TARGETS classes  →  (B, N_BLOCK_TARGETS)
    """

    def __init__(self, d_z: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_z, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_BLOCK_TARGETS),
        )

    def forward(self, z_t: torch.Tensor, z_tn: torch.Tensor) -> torch.Tensor:
        """
        z_t, z_tn : (B, d_z)
        Returns   : (B, N_BLOCK_TARGETS) logits
        """
        x = torch.cat([z_t, z_tn], dim=-1)   # (B, 2 * d_z)
        return self.classifier(x)
