from __future__ import annotations

from crl_vehicle.priors import ConditionalPrior, Prior, StandardPrior
from crl_vehicle.training_modes.base import TrainingMode
from crl_vehicle.training_modes.contrastive_mode import ContrastiveTrainingMode
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
    """Instantiate the TrainingMode implied by config.

    Valid (training_mode, prior_type) combinations:
      ('vae', 'standard')      — classical CRL.
      ('vae', 'conditional')   — iVAE.
      ('contrastive', 'standard') — NT-Xent contrastive (no prior used).
    """
    if config.training_mode == "vae":
        return VAETrainingMode(prior=_build_prior(config), config=config)
    if config.training_mode == "contrastive":
        if config.prior_type != "standard":
            raise ValueError(
                f"training_mode='contrastive' does not use a prior, but "
                f"prior_type={config.prior_type!r} was set explicitly. "
                f"Leave prior_type at 'standard' (the default) for contrastive runs."
            )
        return ContrastiveTrainingMode(config=config)
    raise ValueError(
        f"Unknown training_mode: {config.training_mode!r}. "
        f"Supported: 'vae', 'contrastive'."
    )
