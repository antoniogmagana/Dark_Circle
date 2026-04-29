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
        end_pres = self.D_PRES
        end_type = end_pres + self.D_TYPE
        end_prox = end_type + self.D_PROX
        end_env  = end_prox + self.D_ENV
        self._slices = {
            "pres": slice(0, end_pres),
            "type": slice(end_pres, end_type),
            "prox": slice(end_type, end_prox),
            "env":  slice(end_prox, end_env),
            "free": slice(end_env, d_z),
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


class SplitLatentSpace(nn.Module):
    """Two-block latent: z_signal (vehicle-relevant) ∪ z_env (noise/environment).

    Unlike CausalLatentSpace, makes no claim about which dims encode presence,
    type, or proximity. The signal block is treated as a single labeled
    subspace; routing is done by training losses (cross-modal alignment,
    intervention invariance, temporal stability of env), not by dim assignment.
    """

    def __init__(self, d_z: int = 24, d_signal: int = 12) -> None:
        super().__init__()
        if d_signal <= 0:
            raise ValueError(f"d_signal must be > 0, got {d_signal}")
        if d_signal >= d_z:
            raise ValueError(
                f"d_signal ({d_signal}) must be < d_z ({d_z}) "
                f"to leave room for an env block"
            )
        self.d_z = d_z
        self.d_signal = d_signal
        self.d_env = d_z - d_signal
        self._slices = {
            "signal": slice(0, d_signal),
            "env":    slice(d_signal, d_z),
        }

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return z[..., self._slices["signal"]], z[..., self._slices["env"]]
