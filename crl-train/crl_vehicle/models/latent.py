import torch
import torch.nn as nn


class CausalLatentSpace(nn.Module):
    """
    Named raw-slice split of a d_z=24 latent vector.

    Slots (all raw — no nonlinearities applied here):
        z_pres : dims  0- 3  (4)  vehicle presence
        z_type : dims  4- 9  (6)  vehicle type/class
        z_prox : dims 10-12  (3)  proximity / amplitude
        z_env  : dims 13-18  (6)  environmental / noise factors
        z_free : dims 19-23  (5)  unconstrained remainder

    Downstream heads and auxiliary losses apply their own nonlinearities.
    """

    D_Z    = 24
    D_PRES = 4
    D_TYPE = 6
    D_PROX = 3
    D_ENV  = 6
    D_FREE = 5   # dims 19-23

    def __init__(self, d_z: int = 24):
        super().__init__()
        if d_z != self.D_Z:
            raise ValueError(f"CausalLatentSpace requires d_z={self.D_Z}, got {d_z}")
        self.d_z = d_z

    def split(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (..., 24)
        Returns (z_pres, z_type, z_prox, z_env, z_free) — all raw slices.
        """
        z_pres = z[..., 0:4]
        z_type = z[..., 4:10]
        z_prox = z[..., 10:13]
        z_env  = z[..., 13:19]
        z_free = z[..., 19:24]
        return z_pres, z_type, z_prox, z_env, z_free
