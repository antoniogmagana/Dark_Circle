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
    def __init__(self, d_in: int = CausalLatentSpace.D_TYPE, n_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, z_type: torch.Tensor) -> torch.Tensor:
        return self.head(z_type)


class MLPTypeHead(nn.Module):
    def __init__(
        self,
        d_in: int = CausalLatentSpace.D_TYPE,
        d_hidden: int = 32,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, z_type: torch.Tensor) -> torch.Tensor:
        return self.head(z_type)


class FullZTypeHead(nn.Module):
    def __init__(self, d_z: int, n_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(d_z, n_classes)

    def forward(self, z_full: torch.Tensor) -> torch.Tensor:
        return self.head(z_full)


class LinearProximityHead(nn.Module):
    def __init__(self, d_in: int = CausalLatentSpace.D_PROX) -> None:
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_prox: torch.Tensor) -> torch.Tensor:
        return self.head(z_prox)
