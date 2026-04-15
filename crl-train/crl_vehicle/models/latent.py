import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLatentSpace(nn.Module):
    def __init__(self, d_z=10):
        super().__init__()
        if d_z != 10:
            raise ValueError(f"CausalLatentSpace currently requires d_z=10, got {d_z}")
        self.d_z = d_z

        # dimension sizes
        self.d_pres = 1
        self.d_type = 4
        self.d_prox = 1
        self.d_noise = 4

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z[0], presence, binary
        z_pres = torch.sigmoid(z[..., 0:1])

        # z[1:5] type, categorical
        z_type = F.softmax(z[..., 1:5], dim=-1)

        # z[5] proximity, positive scalar
        z_prox = F.softplus(z[..., 5:6])

        # z[6:10] noise, unconstrained
        z_noise = z[..., 6:10]

        return z_pres, z_type, z_prox, z_noise