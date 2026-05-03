from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class CheckpointState:
    """Mutable state threaded through should_save_checkpoint across epochs.

    Each training mode owns the semantics of its fields. VAETrainingMode
    tracks best_ref_elbo (lower = better) and best_aux_type_f1 (higher =
    better). A contrastive mode would track its own best linear-probe F1.
    Unknown fields are ignored so modes can coexist without schema churn.
    """

    bests: dict[str, float] = field(default_factory=dict)
    best_epochs: dict[str, int] = field(default_factory=dict)
    prev_val_recon: float = float("inf")
    patience_count: int = 0
    # Non-ckpt accounting: last-seen val metrics used by update_beta.
    last_val_metrics: dict[str, float] = field(default_factory=dict)


class TrainingMode(nn.Module, ABC):
    """One coherent CRL training program.

    Subclasses define:
      - forward_pair:          loss + per-batch metrics for one pair batch.
      - val_metrics_summary:   derived metrics (e.g., val_ref_elbo).
      - update_beta:           beta-annealing logic (no-op for non-VAE modes).
      - should_save_checkpoint: which checkpoint files to write this epoch.
      - early_stop_metric:     name of the val metric to monitor for patience.
      - early_stop_mode:       "min" or "max" — which direction improves.

    The Trainer owns the epoch loop, CSV writing, and patience counting; it
    queries the mode for algorithm-specific decisions. Modes are nn.Module
    subclasses because some (iVAE) carry learnable sub-modules like the
    conditional prior MLP.
    """

    @abstractmethod
    def forward_pair(
        self,
        model: nn.Module,
        batch: dict,
        beta: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]: ...

    @abstractmethod
    def val_metrics_summary(self, val_m: dict) -> dict:
        """Augment val metrics with derived fields (e.g., val_ref_elbo)."""
        ...

    @abstractmethod
    def update_beta(
        self, beta: float, val_m: dict, state: CheckpointState, config
    ) -> tuple[float, str]:
        """Return (new_beta, event_string)."""
        ...

    @abstractmethod
    def should_save_checkpoint(
        self, val_m: dict, epoch: int, state: CheckpointState
    ) -> dict[str, bool]:
        """Return {ckpt_filename: should_save_this_epoch}. Mutates state bests."""
        ...

    @abstractmethod
    def early_stop_metric(self) -> str: ...

    @abstractmethod
    def early_stop_mode(self) -> str:
        """Either 'min' or 'max'."""
        ...

    def checkpoint_summary(self, state: CheckpointState) -> dict:
        """Emit the crl_checkpoint_summary.json payload for this mode."""
        return {
            "bests": {k: round(v, 6) for k, v in state.bests.items()},
            "best_epochs": dict(state.best_epochs),
        }

    def set_class_weights(
        self,
        pres_pos_weight: torch.Tensor | None,
        type_class_weights: torch.Tensor | None,
    ) -> None:
        """Inject class weights for any aux supervision the mode applies during CRL.

        Default is a no-op. Modes that compute auxiliary classification losses on
        labels (VAE, disentangled) override this to apply the same weights the
        downstream probe uses, removing the train/eval class-balance mismatch
        that otherwise locks a frozen-backbone downstream into a biased
        representation.
        """
        return None
