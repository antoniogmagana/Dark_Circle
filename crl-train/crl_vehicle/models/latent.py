from __future__ import annotations
import torch
import torch.nn as nn


class CausalLatentSpace(nn.Module):
    D_PRES = 4   # dims 0-3
    D_TYPE = 6   # dims 4-9
    D_PROX = 3   # dims 10-12
    D_ENV  = 6   # dims 13-18
    D_FREE = 5   # dims 19-23
    D_Z    = 24

    def __init__(self, d_z: int = 24) -> None:
        super().__init__()
        if d_z != 24:
            raise ValueError(f"CausalLatentSpace requires d_z=24, got {d_z}")
        self._slices = {
            "pres": slice(0, 4),
            "type": slice(4, 10),
            "prox": slice(10, 13),
            "env":  slice(13, 19),
            "free": slice(19, 24),
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
