"""
VehicleDetectionHead

Downstream heads for vehicle presence detection, type classification, and
instance classification.  Operate on raw (pre-sigmoid/softmax) z slices
from CausalEncoder.split_z_raw().

Design principle: each modality gets its own head; the fusion strategy
(voting, concat, or any-available) is handled in the Trainer.

PresenceHead   : binary detection         — BCEWithLogitsLoss
TypeHead       : 4-class vehicle type     — CrossEntropyLoss
InstanceHead   : 13-class vehicle instance — CrossEntropyLoss
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


class InstanceHead(nn.Module):
    """
    13-class vehicle instance classifier.
    Operates on z_instance (raw logits, shape (B, d_z_instance)).
    Independent of z_type by design — supervised separately.
    """

    def __init__(self, d_z_instance: int = 8, n_instance_classes: int = 13):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_z_instance, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_instance_classes),
        )

    def forward(self, z_instance: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_instance_classes) logits — use CrossEntropyLoss."""
        return self.head(z_instance)


class VehicleDetectionHead(nn.Module):
    """
    Wrapper holding all three heads for a single modality.

    Args:
        d_z_presence      : size of z_presence block
        d_z_type          : size of z_type block
        d_z_instance      : size of z_instance block
        n_classes         : number of vehicle type classes (default 4)
        n_instance_classes: number of vehicle instance classes (default 13)
    """

    def __init__(
        self,
        d_z_presence: int = 1,
        d_z_type: int = 4,
        d_z_instance: int = 8,
        n_classes: int = 4,
        n_instance_classes: int = 13,
    ):
        super().__init__()
        self.presence      = PresenceHead(d_z_presence)
        self.type_head     = TypeHead(d_z_type, n_classes)
        self.instance_head = InstanceHead(d_z_instance, n_instance_classes)

    def forward(
        self,
        z_presence: torch.Tensor,
        z_type: torch.Tensor,
        z_instance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_presence : (B, d_z_presence)   raw presence logit block
        z_type     : (B, d_z_type)       raw type logit block
        z_instance : (B, d_z_instance)   raw instance logit block

        Returns:
            presence_logit  : (B,)               for BCEWithLogitsLoss
            type_logits     : (B, n_classes)      for CrossEntropyLoss
            instance_logits : (B, n_inst_classes) for CrossEntropyLoss
        """
        return (
            self.presence(z_presence),
            self.type_head(z_type),
            self.instance_head(z_instance),
        )
