"""
Linear downstream heads — thin probes on frozen CRL backbone subspaces.

    LinearPresenceHead  : z_pres (B, 4)  → binary detection logit  (B, 1)
    LinearTypeHead      : z_type (B, 6)  → n_classes vehicle logits (B, n_classes)
    LinearProximityHead : z_prox (B, 3)  → proximity scalar         (B, 1)
                          (MSE target = RMS amplitude; range labels when available)
"""

import torch.nn as nn


class LinearPresenceHead(nn.Module):
    def __init__(self, d_in: int = 4):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_pres):
        return self.head(z_pres)   # (B, 1)


class LinearTypeHead(nn.Module):
    def __init__(self, d_in: int = 6, n_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, z_type):
        return self.head(z_type)   # (B, n_classes)


class LinearProximityHead(nn.Module):
    def __init__(self, d_in: int = 3):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_prox):
        return self.head(z_prox)   # (B, 1)
