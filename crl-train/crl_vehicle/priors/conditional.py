from __future__ import annotations

import torch
import torch.nn as nn

from crl_vehicle.config import CATEGORY_TO_IDX, LABEL_MULTI
from crl_vehicle.priors.base import Prior

# Label-space size:
#   LABEL_MULTI     = -2  → index 0
#   LABEL_BACKGROUND = -1 → index 1
#   pedestrian       =  0 → index 2
#   light            =  1 → index 3
#   medium           =  2 → index 4
#   heavy            =  3 → index 5
#
# Offset maps raw labels → non-negative indices so one-hot works directly.
_TYPE_OFFSET = -LABEL_MULTI  # = 2
_N_TYPE_SLOTS = len(CATEGORY_TO_IDX) + _TYPE_OFFSET  # = 6


def _encode_labels(y: torch.Tensor) -> torch.Tensor:
    """y: (B, 2) as (presence_float, type_int_as_float).

    Returns (B, 1 + _N_TYPE_SLOTS) = (B, 7):
      [presence_bit, onehot(type_shifted)]

    Type < LABEL_MULTI is not expected; the offset handles every value
    emitted by the data pipeline. Presence is already {0.0, 1.0}.
    """
    presence = y[:, 0:1].clamp(0.0, 1.0)
    type_int = y[:, 1].long() + _TYPE_OFFSET
    type_int = type_int.clamp(0, _N_TYPE_SLOTS - 1)
    onehot = torch.nn.functional.one_hot(type_int, num_classes=_N_TYPE_SLOTS).float()
    return torch.cat([presence, onehot], dim=-1)


_LABEL_DIM = 1 + _N_TYPE_SLOTS


class ConditionalPrior(Prior):
    """iVAE-style label-conditioned prior p(z | y) = N(μ(y), σ²(y)).

    Replaces StandardPrior's fixed N(0, I) with a learned, label-conditioned
    diagonal Gaussian. Identifiability: under sufficient label variation,
    the latent z is identifiable up to component-wise transformation —
    a theoretical guarantee StandardPrior does not provide.

    Architecture: small MLP (presence_onehot + type_onehot) → (μ, log σ²).
    Weights initialize small so the prior starts ≈ N(0, I) and diverges as
    training progresses. logvar is clamped to prevent the prior from
    becoming degenerate (near-zero variance collapses KL).
    """

    LOGVAR_MIN = -4.0
    LOGVAR_MAX = 4.0

    def __init__(
        self,
        d_z: int,
        hidden_dim: int = 32,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(_LABEL_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_z),
        )
        # Small-init: last layer near zero so prior starts at ≈ N(0, I).
        # Without this, the prior MLP would output random (μ, logvar) from
        # the start, giving the encoder a moving target before training
        # even begins.
        with torch.no_grad():
            self.net[-1].weight.mul_(init_scale)
            self.net[-1].bias.zero_()

    def prior_params(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 2) labels → ((B, d_z) prior_mu, (B, d_z) prior_logvar)."""
        h = _encode_labels(y)
        out = self.net(h)
        mu_p, logvar_p = out.chunk(2, dim=-1)
        logvar_p = logvar_p.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        return mu_p, logvar_p

    def kl_to_posterior(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError(
                "ConditionalPrior requires auxiliary labels y. "
                "VAETrainingMode threads y=(presence, type) through _kl_terms; "
                "if you hit this, check that the caller is passing y."
            )
        mu_p, logvar_p = self.prior_params(y)

        # Analytical KL[N(μ_q, σ_q²) || N(μ_p, σ_p²)] for diagonal Gaussians,
        # per dim:
        #   0.5 * (log σ_p² - log σ_q² + (σ_q² + (μ_q - μ_p)²) / σ_p² - 1)
        var_q = logvar.exp()
        inv_var_p = (-logvar_p).exp()
        kl = 0.5 * (logvar_p - logvar + (var_q + (mu - mu_p).pow(2)) * inv_var_p - 1.0)
        return kl.sum(dim=-1).mean()
