"""
MultiModalCausalVAE

Architecture:
  - One ModalityEncoder (1D CNN) per modality (audio, seismic, accel)
  - Masked mean-pool fusion with a learnable availability embedding
  - Split latent space:
      z_veh : vehicle identity — deterministic projection, trained with
              VICReg-style invariance/variance/covariance objectives.
              No logvar, no KL, no decoder.
      z_env : sensor/environment — VAE with iVAE conditional prior keyed
              by sensor domain ID and temporal slowness regulariser.
  - No waveform decoder: reconstruction was a degenerate objective because
    upsampled seismic windows are near-identical, making MSE trivially small
    regardless of what the encoder learns.

The model supports missing modalities: if a modality is absent, its
contribution to the fused representation is zeroed out. An availability
embedding informs the model which subset of modalities is present.
"""

import torch
import torch.nn as nn
from crl_config import MODALITIES, MODALITY_CHANNELS


# ---------------------------------------------------------------------------
# Modality encoder
# ---------------------------------------------------------------------------


class ModalityEncoder(nn.Module):
    """
    1D CNN encoder for a single sensor modality.

    Input : [B, C_in, T]
    Output: [B, out_dim]
    """

    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


# ---------------------------------------------------------------------------
# Availability embedding
# ---------------------------------------------------------------------------


def _availability_idx(availability: torch.Tensor) -> torch.Tensor:
    """
    Map a [B, num_modalities] bool tensor to a scalar index per sample.
    Treats the availability row as a binary number (MSB = first modality).
    With 3 modalities there are 8 possible subsets (0-7).
    """
    n = availability.shape[1]
    powers = torch.tensor(
        [2 ** (n - 1 - i) for i in range(n)],
        dtype=torch.long,
        device=availability.device,
    )
    return (availability.long() * powers).sum(dim=1)  # [B]


# ---------------------------------------------------------------------------
# MultiModalCausalVAE
# ---------------------------------------------------------------------------


class MultiModalCausalVAE(nn.Module):
    """
    Args:
        num_sensor_domains : number of unique (dataset, rs_node) pairs
        modality_feat_dim  : output dim of each ModalityEncoder (default 128)
        z_veh_dim          : vehicle latent dimension
        z_env_dim          : environment latent dimension
    """

    def __init__(
        self,
        num_sensor_domains: int,
        modality_feat_dim: int = 128,
        z_veh_dim: int = 32,
        z_env_dim: int = 16,
    ):
        super().__init__()
        self.z_veh_dim = z_veh_dim
        self.z_env_dim = z_env_dim
        self.feat_dim = modality_feat_dim

        # Per-modality encoders
        self.encoders = nn.ModuleDict(
            {
                mod: ModalityEncoder(
                    in_channels=MODALITY_CHANNELS[mod],
                    out_dim=modality_feat_dim,
                )
                for mod in MODALITIES
            }
        )

        # Availability embedding: 2^num_modalities possible subsets
        num_subsets = 2 ** len(MODALITIES)
        self.avail_embed = nn.Embedding(num_subsets, modality_feat_dim)

        # z_veh: deterministic projection (VICReg, no KL)
        self.fc_veh = nn.Linear(modality_feat_dim, z_veh_dim)

        # z_env: VAE projection
        self.fc_mu_env = nn.Linear(modality_feat_dim, z_env_dim)
        self.fc_logvar_env = nn.Linear(modality_feat_dim, z_env_dim)

        # iVAE conditional prior for z_env, conditioned on sensor domain
        self.env_prior_mu = nn.Embedding(num_sensor_domains, z_env_dim)
        self.env_prior_logvar = nn.Embedding(num_sensor_domains, z_env_dim)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _fuse(
        self,
        batch_mods: dict,
        availability: torch.Tensor,
    ) -> tuple:
        """
        Encode all available modalities and fuse them.

        Args:
            batch_mods   : {mod: Tensor [B, C, T] or None}
            availability : bool Tensor [B, num_modalities]

        Returns:
            z_veh      [B, z_veh_dim]  — deterministic vehicle embedding
            mu_env     [B, z_env_dim]
            logvar_env [B, z_env_dim]
        """
        B = availability.shape[0]
        feat_sum = torch.zeros(B, self.feat_dim, device=availability.device)
        avail_count = availability.float().sum(dim=1, keepdim=True).clamp(min=1.0)

        for i, mod in enumerate(MODALITIES):
            x = batch_mods.get(mod)
            if x is None:
                continue
            mask = availability[:, i].float().unsqueeze(1)  # [B, 1]
            feat = self.encoders[mod](x)  # [B, feat_dim]
            feat_sum = feat_sum + feat * mask

        pooled = feat_sum / avail_count  # masked mean [B, feat_dim]

        # Condition on which modalities are available
        avail_idx = _availability_idx(availability)  # [B]
        pooled = pooled + self.avail_embed(avail_idx)  # [B, feat_dim]

        z_veh = self.fc_veh(pooled)
        mu_env = self.fc_mu_env(pooled)
        logvar_env = self.fc_logvar_env(pooled).clamp(min=-4.0, max=4.0)
        return z_veh, mu_env, logvar_env

    def encode_veh(
        self,
        batch_mods: dict,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return z_veh for use by downstream heads.
        Encoder weights should be frozen when calling this.
        """
        z_veh, _, _ = self._fuse(batch_mods, availability)
        return z_veh

    # ------------------------------------------------------------------
    # Reparameterization (z_env only)
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch_t: dict,
        batch_next: dict,
        availability: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> dict:
        """
        Full forward pass for CRL pre-training.

        Returns a dict with keys needed for loss computation:
          z_veh_t          : [B, z_veh_dim]  deterministic vehicle embedding at t
          z_veh_next       : [B, z_veh_dim]  deterministic vehicle embedding at t+1
          mu_env_t         : [B, z_env_dim]
          logvar_env_t     : [B, z_env_dim]
          z_env_t          : [B, z_env_dim]  reparameterised
          z_env_next       : [B, z_env_dim]  mean only, for slow loss
          prior_mu_env     : [B, z_env_dim]
          prior_logvar_env : [B, z_env_dim]
        """
        # Encode t
        z_veh_t, mu_env_t, logvar_env_t = self._fuse(batch_t, availability)
        z_env_t = self.reparameterize(mu_env_t, logvar_env_t)

        # Encode t+1 (z_env mean only for slow loss; z_veh for VICReg invariance)
        z_veh_next, mu_env_next, _ = self._fuse(batch_next, availability)
        z_env_next = mu_env_next

        # iVAE prior for z_env conditioned on sensor domain
        prior_mu_env = self.env_prior_mu(domain_ids)
        prior_logvar_env = self.env_prior_logvar(domain_ids).clamp(min=-4.0, max=4.0)

        return {
            "z_veh_t": z_veh_t,
            "z_veh_next": z_veh_next,
            "mu_env_t": mu_env_t,
            "logvar_env_t": logvar_env_t,
            "z_env_t": z_env_t,
            "z_env_next": z_env_next,
            "prior_mu_env": prior_mu_env,
            "prior_logvar_env": prior_logvar_env,
        }
