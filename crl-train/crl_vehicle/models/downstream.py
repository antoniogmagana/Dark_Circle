"""
VehicleDetectionHead

Downstream heads for vehicle presence detection and type classification.
Operate on the raw (pre-sigmoid/softmax) z slices from CausalEncoder.split_z_raw().

Design principle: each modality gets its own head; the fusion strategy
(voting, concat, or any-available) is handled in the Trainer.

PresenceHead   : binary detection    — BCEWithLogitsLoss
TypeHead       : 4-class vehicle type — CrossEntropyLoss
"""

import torch
import torch.nn as nn


class PresenceHead(nn.Module):
    """
    Binary vehicle presence classifier.
    Operates on z_presence (raw logit, shape (B, d_z_presence)).

    Single linear layer — the latent block should already encode presence
    after CRL pre-training; a deeper head risks overfitting the linear probe.
    """

    def __init__(self, d_z_presence: int = 1):
        super().__init__()
        self.head = nn.Linear(d_z_presence, 1)

    def forward(self, z_presence: torch.Tensor) -> torch.Tensor:
        """Returns (B,) logit — use BCEWithLogitsLoss."""
        return self.head(z_presence).squeeze(-1)


class TypeHead(nn.Module):
    """
    4-class vehicle type classifier.
    Operates on z_type (raw logits, shape (B, d_z_type)).

    One hidden layer to allow mild non-linearity while remaining a
    "nearly linear" probe during the linear probe evaluation phase.
    """

    def __init__(self, d_z_type: int = 4, n_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_z_type, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )

    def forward(self, z_type: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_classes) logits — use CrossEntropyLoss."""
        return self.head(z_type)


class VehicleDetectionHead(nn.Module):
    """
    Wrapper holding both heads for a single modality.

    Args:
        d_z_presence : size of z_presence block
        d_z_type     : size of z_type block
        n_classes    : number of vehicle type classes (default 4)
    """

    def __init__(
        self,
        d_z_presence: int = 1,
        d_z_type: int = 4,
        n_classes: int = 4,
    ):
        super().__init__()
        self.presence = PresenceHead(d_z_presence)
        self.type_head = TypeHead(d_z_type, n_classes)

    def forward(
        self, z_presence: torch.Tensor, z_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_presence : (B, d_z_presence)  raw presence logit block
        z_type     : (B, d_z_type)      raw type logit block

        Returns:
            presence_logit : (B,)          for BCEWithLogitsLoss
            type_logits    : (B, n_classes) for CrossEntropyLoss
        """
        return self.presence(z_presence), self.type_head(z_type)
