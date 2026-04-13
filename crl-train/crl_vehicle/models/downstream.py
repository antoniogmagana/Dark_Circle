"""
VehicleDetectionHead

Downstream heads for vehicle presence detection, type classification, and
instance classification. Operate on task-specific embeddings produced by
MultiTaskEncoder — one embedding per task, per modality.

Design principle: each modality gets its own head; the fusion strategy
(voting or any-available) is handled in the Trainer.

PresenceHead   : binary detection         — BCEWithLogitsLoss
TypeHead       : 4-class vehicle type     — CrossEntropyLoss
InstanceHead   : 13-class vehicle instance — CrossEntropyLoss
"""

import torch
import torch.nn as nn


class PresenceHead(nn.Module):
    """
    Binary vehicle presence classifier.
    Operates on e_pres (shape (B, d_pres)).

    Single linear layer — the presence embedding should already encode
    presence after CRL pre-training; a deeper head risks overfitting.
    """

    def __init__(self, d_pres: int = 16):
        super().__init__()
        self.head = nn.Linear(d_pres, 1)

    def forward(self, e_pres: torch.Tensor) -> torch.Tensor:
        """Returns (B,) logit — use BCEWithLogitsLoss."""
        return self.head(e_pres).squeeze(-1)


class TypeHead(nn.Module):
    """
    4-class vehicle type classifier.
    Operates on e_type (shape (B, d_type)).
    """

    def __init__(self, d_type: int = 32, n_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_type, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )

    def forward(self, e_type: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_classes) logits — use CrossEntropyLoss."""
        return self.head(e_type)


class InstanceHead(nn.Module):
    """
    13-class vehicle instance classifier.
    Operates on e_inst (shape (B, d_inst)).
    """

    def __init__(self, d_inst: int = 64, n_instance_classes: int = 13):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_inst, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_instance_classes),
        )

    def forward(self, e_inst: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_instance_classes) logits — use CrossEntropyLoss."""
        return self.head(e_inst)


class VehicleDetectionHead(nn.Module):
    """
    Wrapper holding all three downstream heads for a single modality.

    Args:
        d_pres            : presence embedding dimension
        d_type            : type embedding dimension
        d_inst            : instance embedding dimension
        n_classes         : number of vehicle type classes (default 4)
        n_instance_classes: number of vehicle instance classes (default 13)
    """

    def __init__(
        self,
        d_pres: int = 16,
        d_type: int = 32,
        d_inst: int = 64,
        n_classes: int = 4,
        n_instance_classes: int = 13,
    ):
        super().__init__()
        self.presence      = PresenceHead(d_pres)
        self.type_head     = TypeHead(d_type, n_classes)
        self.instance_head = InstanceHead(d_inst, n_instance_classes)

    def forward(
        self,
        e_pres: torch.Tensor,
        e_type: torch.Tensor,
        e_inst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        e_pres: (B, d_pres)  presence embedding
        e_type: (B, d_type)  type embedding
        e_inst: (B, d_inst)  instance embedding

        Returns:
            presence_logit  : (B,)               for BCEWithLogitsLoss
            type_logits     : (B, n_classes)      for CrossEntropyLoss
            instance_logits : (B, n_inst_classes) for CrossEntropyLoss
        """
        return (
            self.presence(e_pres),
            self.type_head(e_type),
            self.instance_head(e_inst),
        )
