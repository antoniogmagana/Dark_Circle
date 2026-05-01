"""Prior distributions for the CRL latent space.

The Prior abstraction lets KL regularization swap between:
  - StandardPrior:  N(0, I) — the classical VAE prior.
  - ConditionalPrior (Checkpoint 2): N(μ(y), σ²(y)) — iVAE-style
    label-conditioned prior that gives identifiability under
    auxiliary label variation.

All Prior subclasses expose kl_to_posterior(mu, logvar, y) and return a
scalar tensor (summed over latent dims, meaned over batch, unscaled by beta —
beta application lives in the training mode).
"""

from crl_vehicle.priors.base import Prior
from crl_vehicle.priors.conditional import ConditionalPrior
from crl_vehicle.priors.standard import StandardPrior

__all__ = ["Prior", "StandardPrior", "ConditionalPrior"]
