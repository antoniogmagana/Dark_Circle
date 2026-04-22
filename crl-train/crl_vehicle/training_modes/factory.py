from __future__ import annotations

from crl_vehicle.priors import ConditionalPrior, Prior, StandardPrior
from crl_vehicle.training_modes.base import TrainingMode
from crl_vehicle.training_modes.vae_mode import VAETrainingMode


def _build_prior(config) -> Prior:
    prior_type = config.prior_type
    if prior_type == "standard":
        return StandardPrior()
    if prior_type == "conditional":
        return ConditionalPrior(d_z=config.d_z)
    raise ValueError(
        f"Unknown prior_type: {prior_type!r}. "
        f"Supported: 'standard', 'conditional'."
    )


def build_training_mode(config) -> TrainingMode:
    """Instantiate the TrainingMode + Prior pair implied by config.

    Valid (training_mode, prior_type) combinations:
      ('vae', 'standard')    — classical CRL.
      ('vae', 'conditional') — iVAE.
      ('contrastive', *)     — Checkpoint 3.
    """
    prior = _build_prior(config)
    if config.training_mode == "vae":
        return VAETrainingMode(prior=prior, config=config)
    raise ValueError(
        f"Unknown training_mode: {config.training_mode!r}. "
        f"Supported: 'vae'. "
        f"Contrastive arrives in Checkpoint 3."
    )
