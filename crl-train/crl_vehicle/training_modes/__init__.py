"""Training mode abstractions for CRL pre-training.

A TrainingMode encapsulates the forward pass, loss computation, beta schedule,
and checkpoint-selection logic for one coherent training program. The Trainer
class dispatches to whichever mode is configured; the Trainer itself contains
no algorithm-specific logic.

At Checkpoint 1 only VAETrainingMode is implemented — it reproduces the
existing CRL pipeline (recon + KL + intervention + aux losses, dual-checkpoint
selection by val_ref_elbo and val_aux_type_f1). Future modes:
  - ContrastiveTrainingMode (Checkpoint 3): NT-Xent over StratifiedPairDataset.
  - others as needed.

Selection is via CRLConfig.training_mode. Factory function build_training_mode
reads the config and instantiates the correct mode + prior.
"""

from crl_vehicle.training_modes.base import CheckpointState, TrainingMode
from crl_vehicle.training_modes.contrastive_mode import ContrastiveTrainingMode
from crl_vehicle.training_modes.disentangled_mode import DisentangledVAETrainingMode
from crl_vehicle.training_modes.factory import build_training_mode
from crl_vehicle.training_modes.vae_mode import VAETrainingMode

__all__ = [
    "TrainingMode",
    "VAETrainingMode",
    "ContrastiveTrainingMode",
    "DisentangledVAETrainingMode",
    "CheckpointState",
    "build_training_mode",
]
