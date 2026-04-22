from __future__ import annotations

from crl_vehicle.priors import Prior, StandardPrior
from crl_vehicle.training_modes.base import TrainingMode
from crl_vehicle.training_modes.vae_mode import VAETrainingMode


def _build_prior(prior_type: str) -> Prior:
    if prior_type == "standard":
        return StandardPrior()
    raise ValueError(
        f"Unknown prior_type: {prior_type!r}. "
        f"Supported: 'standard'. "
        f"ConditionalPrior ('conditional') arrives in Checkpoint 2."
    )


def build_training_mode(config) -> TrainingMode:
    """Instantiate the TrainingMode + Prior pair implied by config.

    Valid (training_mode, prior_type) combinations:
      ('vae', 'standard')    — classical CRL. This is Checkpoint 1.
      ('vae', 'conditional') — iVAE. Lands in Checkpoint 2.
      ('contrastive', *)     — Checkpoint 3.
    """
    prior = _build_prior(config.prior_type)
    if config.training_mode == "vae":
        return VAETrainingMode(prior=prior, config=config)
    raise ValueError(
        f"Unknown training_mode: {config.training_mode!r}. "
        f"Supported at Checkpoint 1: 'vae'. "
        f"Contrastive arrives in Checkpoint 3."
    )
