"""
Linear downstream heads.

These are thin linear probes trained on a frozen CRL backbone.
The latent space is expected to do the representational work — heads are
intentionally kept to a single linear layer with no hidden layers.

LinearPresenceHead : z_pres (B, 1)  → binary detection logit  (B, 1)
LinearTypeHead     : z_type (B, 4)  → 4-class vehicle logits  (B, 4)
"""

import torch.nn as nn


class LinearPresenceHead(nn.Module):
    def __init__(self, d_in: int = 1):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_pres):
        # z_pres: (B, 1) → (B, 1)
        return self.head(z_pres)


class LinearTypeHead(nn.Module):
    def __init__(self, d_in: int = 4, n_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, z_type):
        # z_type: (B, 4) → (B, n_classes)
        return self.head(z_type)
