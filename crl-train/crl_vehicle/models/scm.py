"""
SCM — Structural Causal Model over the latent space z.

Learns a weighted DAG adjacency matrix A (d_z × d_z) and a per-variable
causal mechanism f_i.  The NOTEARS acyclicity constraint h(A) = 0 is
enforced as a soft penalty during training.

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
        # Initialised near zero → sparse graph at the start of training.
        self.A_raw = nn.Parameter(
            torch.zeros(d_z, d_z) + 0.01 * torch.randn(d_z, d_z)
        )

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
        Soft adjacency matrix with diagonal forced to 0 (no self-loops).
        Returns (d_z, d_z) with values in (0, 1).
        """
        A = torch.sigmoid(self.A_raw)
        diag_mask = torch.eye(self.d_z, device=A.device)
        return A * (1.0 - diag_mask)

    def acyclicity_loss(self) -> torch.Tensor:
        """
        NOTEARS constraint: h(A) = tr(e^{A ∘ A}) - d = 0 iff A is a DAG.
        Returns a scalar; = 0 for a perfect DAG, > 0 if cycles remain.
        Gradient clipping (max_norm=1.0 in the trainer) is important here
        since this term can spike early when the graph is still cyclic.
        """
        A = self.adjacency()
        M = A * A                                        # element-wise square
        if A.device.type == "mps":
            E = torch.linalg.matrix_exp(M.cpu()).to(A.device)  # MPS lacks this op
        else:
            E = torch.linalg.matrix_exp(M)
        return torch.trace(E) - self.d_z                # scalar

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
