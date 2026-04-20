from __future__ import annotations
import torch
import torch.nn as nn


class CausalLatentSpace(nn.Module):
    D_PRES = 4   # dims 0-3
    D_TYPE = 6   # dims 4-9
    D_PROX = 3   # dims 10-12
    D_ENV  = 6   # dims 13-18
    D_CAUSAL = D_PRES + D_TYPE + D_PROX + D_ENV  # = 19

    def __init__(self, d_z: int = 24) -> None:
        super().__init__()
        if d_z <= self.D_CAUSAL:
            raise ValueError(
                f"CausalLatentSpace requires d_z > {self.D_CAUSAL} "
                f"(to leave room for a free subspace), got {d_z}"
            )
        self.d_z = d_z
        self.d_free = d_z - self.D_CAUSAL
        self._slices = {
            "pres": slice(0, 4),
            "type": slice(4, 10),
            "prox": slice(10, 13),
            "env":  slice(13, 19),
            "free": slice(19, d_z),
        }

    def split(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            z[..., self._slices["pres"]],
            z[..., self._slices["type"]],
            z[..., self._slices["prox"]],
            z[..., self._slices["env"]],
            z[..., self._slices["free"]],
        )
