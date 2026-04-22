from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Prior(nn.Module, ABC):
    """Abstract prior over the latent space.

    Training modes call ``prior.kl_to_posterior(mu, logvar, y)`` to compute
    a scalar KL[q(z|x) || p(z|y)], summed over latent dims and meaned over
    batch. The returned value is not multiplied by beta — beta is applied
    by the training mode after selecting a prior.

    y carries auxiliary labels (e.g., (B, 2) of [presence, type]). Priors
    that do not condition on labels (StandardPrior) ignore y. Plumbing y
    through at Checkpoint 1 — even though StandardPrior discards it — keeps
    iVAE (Checkpoint 2) a pure additive change.
    """

    @abstractmethod
    def kl_to_posterior(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
