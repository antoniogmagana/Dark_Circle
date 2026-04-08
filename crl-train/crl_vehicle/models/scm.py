"""
SCM — Structural Causal Model over the latent space z.

Learns a weighted DAG adjacency matrix A (d_z × d_z) and a per-variable
causal mechanism f_i.  Acyclicity is enforced by construction via strict
lower-triangular parameterization — no penalty term required.

One SCM instance is shared across modalities (same causal graph).
"""

import torch
import torch.nn as nn


class SCM(nn.Module):
    """
    Args:
        d_z        : total latent dimension
        hidden_dim : hidden size of each per-variable MLP mechanism
    """

    def __init__(self, d_z: int, hidden_dim: int = 32):
        super().__init__()
        self.d_z = d_z

        # Learnable adjacency matrix A_raw (d_z × d_z).
        # A[i, j] = weight of edge z_j → z_i.
        # Only the strict lower triangle is used; upper triangle and diagonal
        # are zeroed out, guaranteeing a DAG by construction.
        self.A_raw = nn.Parameter(torch.randn(d_z, d_z) * 0.3)

        # Per-variable causal mechanisms: f_i : R^d_z → R
        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_z, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(d_z)
        ])

    def adjacency(self) -> torch.Tensor:
        """
        Strict lower-triangular adjacency matrix — acyclic by construction.
        A[i, j] = weight of edge z_j → z_i (j < i only).
        """
        return torch.tril(self.A_raw, diagonal=-1)

    def forward(
        self,
        z: torch.Tensor,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict z_hat from the causal mechanisms given parent values.

        z                : (B, d_z) — encoder-sampled latent
        intervention_mask: (B, d_z) binary, 1 = externally intervened;
                           intervened variables take their encoder value,
                           not the SCM mechanism output.

        Returns z_hat: (B, d_z)
        """
        A = self.adjacency()   # (d_z, d_z)
        z_hat = torch.zeros_like(z)

        for i, mech in enumerate(self.mechanisms):
            # Weighted parent values for variable i
            parents = A[i].unsqueeze(0) * z   # (B, d_z)
            z_hat[:, i] = mech(parents).squeeze(-1)

        if intervention_mask is not None:
            # Intervened variables keep their encoder-sampled value
            mask = intervention_mask.float()
            z_hat = z_hat * (1.0 - mask) + z * mask

        return z_hat
