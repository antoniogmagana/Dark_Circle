from __future__ import annotations
import torch
import torch.nn as nn
from crl_vehicle.models.latent import CausalLatentSpace


class LinearPresenceHead(nn.Module):
    def __init__(self, d_in: int = CausalLatentSpace.D_PRES) -> None:
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_pres: torch.Tensor) -> torch.Tensor:
        return self.head(z_pres)


class LinearTypeHead(nn.Module):
    def __init__(
        self, d_in: int = CausalLatentSpace.D_TYPE, n_classes: int = 4
    ) -> None:
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, z_type: torch.Tensor) -> torch.Tensor:
        return self.head(z_type)


class LinearProximityHead(nn.Module):
    def __init__(self, d_in: int = CausalLatentSpace.D_PROX) -> None:
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_prox: torch.Tensor) -> torch.Tensor:
        return self.head(z_prox)
