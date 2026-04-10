"""
SpectralDecoder

Reconstructs filterbank log-envelope output from the latent z.
Target shape: (B, K, T') — same as filterbank output.

Reconstructing envelopes (not raw waveforms) avoids high-frequency phase
ambiguity and provides a tractable ELBO signal.  One decoder per modality.
"""

import torch
import torch.nn as nn

from crl_vehicle.config import CRLConfig, ModalityConfig


class SpectralDecoder(nn.Module):
    """
    Args:
        d_z     : latent dimension (CRLConfig.d_z)
        d_model : hidden dimension
        mod_cfg : ModalityConfig for the target modality
                  (provides filterbank_out_channels and t_prime)
    """

    def __init__(self, d_z: int, d_model: int, mod_cfg: ModalityConfig):
        super().__init__()
        self.K = mod_cfg.filterbank_out_channels
        self.T = mod_cfg.t_prime

        self.expand = nn.Linear(d_z, d_model)
        self.decode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.K * self.T),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z   : (B, d_z)
        Returns: (B, K, T')
        """
        h = torch.relu(self.expand(z))    # (B, d_model)
        out = self.decode(h)              # (B, K*T')
        return out.view(z.shape[0], self.K, self.T)
