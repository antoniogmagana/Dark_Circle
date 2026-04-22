from __future__ import annotations

import torch

from crl_vehicle.priors.base import Prior


class StandardPrior(Prior):
    """N(0, I) prior — classical VAE.

    KL[q(z|x) || N(0, I)] = 0.5 * sum(var + mu^2 - 1 - log(var)),
    meaned over batch. Ignores y.
    """

    def kl_to_posterior(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=-1)
        return kl.mean()
