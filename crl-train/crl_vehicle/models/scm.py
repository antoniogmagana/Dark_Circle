"""
SCM — Structural Causal Model over the latent space z.

Learns a weighted DAG adjacency matrix A (d_z × d_z) and a per-variable
causal mechanism f_i.  Acyclicity is enforced by construction via strict
lower-triangular parameterization — no penalty term required.

One SCM instance is shared across modalities (same causal graph).

Mechanisms are implemented as a single batched einsum over all d_z variables
rather than a Python loop, eliminating d_z sequential kernel launches.
"""

import math

import torch
import torch.nn as nn


class SCM(nn.Module):
    """
    Args:
        d_z        : total latent dimension
        hidden_dim : hidden size of each per-variable MLP mechanism
    """

    def __init__(self, d_z: int, hidden_dim: int = 32, group_sizes: list | None = None):
        super().__init__()
        self.d_z = d_z
        self.hidden_dim = hidden_dim

        # Learnable adjacency matrix A_raw (d_z × d_z).
        # A[i, j] = weight of edge z_j → z_i.
        # Only the strict lower triangle is used; upper triangle and diagonal
        # are zeroed out, guaranteeing a DAG by construction.
        self.A_raw = nn.Parameter(torch.randn(d_z, d_z) * 0.3)

        # Static edge mask: controls which causal edges are permitted.
        # group_sizes = [d_z_presence, d_z_type, d_z_instance, d_z_proximity, d_z_noise]
        # Permitted edges: z_type → z_instance only.  All other groups are
        # treated as independent roots (no intra-group or cross-group edges).
        if group_sizes is not None:
            mask = torch.zeros(d_z, d_z)
            starts = [0]
            for s in group_sizes:
                starts.append(starts[-1] + s)
            # group 1 = type, group 2 = instance
            type_start, type_end = starts[1], starts[2]
            inst_start, inst_end = starts[2], starts[3]
            mask[inst_start:inst_end, type_start:type_end] = 1.0
            mask = torch.tril(mask, diagonal=-1)
        else:
            mask = torch.tril(torch.ones(d_z, d_z), diagonal=-1)
        self.register_buffer("edge_mask", mask)

        # Per-variable causal mechanisms: f_i : R^d_z → R
        # Stored as batched weight tensors to allow a single vectorised forward
        # pass instead of d_z sequential kernel launches.
        #   mech_W1 : (d_z, d_z, hidden_dim)  — first linear weights
        #   mech_b1 : (d_z, hidden_dim)        — first linear biases
        #   mech_W2 : (d_z, hidden_dim)        — second linear weights (scalar out)
        #   mech_b2 : (d_z,)                   — output biases
        self.mech_W1 = nn.Parameter(
            torch.randn(d_z, d_z, hidden_dim) / math.sqrt(d_z)
        )
        self.mech_b1 = nn.Parameter(torch.zeros(d_z, hidden_dim))
        self.mech_W2 = nn.Parameter(
            torch.randn(d_z, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.mech_b2 = nn.Parameter(torch.zeros(d_z))

    def adjacency(self) -> torch.Tensor:
        """
        Block-masked lower-triangular adjacency matrix — acyclic by construction.
        Only z_type → z_instance edges are permitted; all other groups are
        independent roots.  A[i, j] = weight of edge z_j → z_i (j < i only).
        """
        return torch.tril(self.A_raw, diagonal=-1) * self.edge_mask

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
        A = self.adjacency()  # (d_z, d_z)

        # Weighted parent inputs for all variables simultaneously.
        # inputs[b, i, j] = A[i, j] * z[b, j]
        inputs = A.unsqueeze(0) * z.unsqueeze(1)  # (B, d_z, d_z)

        # First layer: (B, d_z, hidden_dim)
        h1 = torch.tanh(
            torch.einsum("bij,ijh->bih", inputs, self.mech_W1) + self.mech_b1
        )

        # Second layer (scalar output per variable): (B, d_z)
        z_hat = torch.einsum("bih,ih->bi", h1, self.mech_W2) + self.mech_b2

        if intervention_mask is not None:
            mask = intervention_mask.float()
            z_hat = z_hat * (1.0 - mask) + z * mask

        return z_hat
