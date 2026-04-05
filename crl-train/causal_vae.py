"""
MultiModalCausalVAE

Architecture:
  - One ModalityEncoder (1D CNN) per modality (audio, seismic, accel)
  - Masked mean-pool fusion with a learnable availability embedding
  - Shared latent space: z_veh (vehicle identity) + z_env (sensor/environment)
  - iVAE conditional prior on z_env keyed by sensor domain ID
  - Per-modality decoders for reconstruction
  - Slow-feature regulariser applies to z_env only

The model supports missing modalities: if a modality is absent, its
contribution to the fused representation is zeroed out. An availability
embedding informs the model which subset of modalities is present.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from crl_config import MODALITIES, MODALITY_CHANNELS, REF_SR, SAMPLE_SECONDS


# ---------------------------------------------------------------------------
# Modality encoder / decoder
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


class ModalityDecoder(nn.Module):
    """
    1D CNN decoder (mirror of ModalityEncoder).

    Input : [B, z_total]
    Output: [B, C_out, T_out]  where T_out ≈ REF_SR * SAMPLE_SECONDS
    """

    def __init__(self, z_total: int, out_channels: int, out_len: int):
        super().__init__()
        self.out_len = out_len
        self.stem = nn.Linear(z_total, 128 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=4, padding=0),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z).view(z.size(0), 128, 8)
        x = self.net(h)
        # Trim or pad to target length
        if x.size(-1) > self.out_len:
            x = x[..., :self.out_len]
        elif x.size(-1) < self.out_len:
            x = F.pad(x, (0, self.out_len - x.size(-1)))
        return x


# ---------------------------------------------------------------------------
# Availability embedding
# ---------------------------------------------------------------------------

def _availability_idx(availability: torch.Tensor) -> torch.Tensor:
    """
    Map a [B, num_modalities] bool tensor to a scalar index per sample.
    Treats the availability row as a binary number (MSB = first modality).

    With 3 modalities there are 8 possible subsets (0–7).
    """
    n = availability.shape[1]
    powers = torch.tensor(
        [2 ** (n - 1 - i) for i in range(n)],
        dtype=torch.long,
        device=availability.device,
    )
    return (availability.long() * powers).sum(dim=1)   # [B]


# ---------------------------------------------------------------------------
# MultiModalCausalVAE
# ---------------------------------------------------------------------------

class MultiModalCausalVAE(nn.Module):
    """
    Args:
        modality_feat_dim  : output dim of each ModalityEncoder (default 128)
        num_sensor_domains : number of unique (dataset, rs_node) pairs
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
        self.z_total   = z_veh_dim + z_env_dim
        self.feat_dim  = modality_feat_dim

        out_len = int(REF_SR * SAMPLE_SECONDS)

        # Per-modality encoders and decoders
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(
                in_channels=MODALITY_CHANNELS[mod],
                out_dim=modality_feat_dim,
            )
            for mod in MODALITIES
        })
        # Decoders take only z_veh — reconstruction loss must flow through
        # z_veh, preventing the decoder from using z_env as a shortcut.
        self.decoders = nn.ModuleDict({
            mod: ModalityDecoder(
                z_total=self.z_veh_dim,
                out_channels=MODALITY_CHANNELS[mod],
                out_len=out_len,
            )
            for mod in MODALITIES
        })

        # Availability embedding: 2^num_modalities possible subsets
        num_subsets = 2 ** len(MODALITIES)
        self.avail_embed = nn.Embedding(num_subsets, modality_feat_dim)

        # Fusion projection: (pooled features + availability embed) → z parameters
        self.fc_mu     = nn.Linear(modality_feat_dim, self.z_total)
        self.fc_logvar = nn.Linear(modality_feat_dim, self.z_total)

        # iVAE conditional prior for z_env, conditioned on sensor domain
        self.env_prior_mu     = nn.Embedding(num_sensor_domains, z_env_dim)
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
            mu [B, z_total], logvar [B, z_total]
        """
        B = availability.shape[0]
        feat_sum  = torch.zeros(B, self.feat_dim, device=availability.device)
        avail_count = availability.float().sum(dim=1, keepdim=True).clamp(min=1.0)

        for i, mod in enumerate(MODALITIES):
            x = batch_mods.get(mod)
            if x is None:
                continue
            mask = availability[:, i].float().unsqueeze(1)  # [B, 1]
            feat = self.encoders[mod](x)                     # [B, feat_dim]
            feat_sum = feat_sum + feat * mask

        pooled = feat_sum / avail_count   # masked mean  [B, feat_dim]

        # Condition on which modalities are available
        avail_idx = _availability_idx(availability)          # [B]
        pooled = pooled + self.avail_embed(avail_idx)        # [B, feat_dim]

        mu     = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled).clamp(min=-4.0, max=4.0)
        return mu, logvar

    def encode(
        self,
        batch_mods: dict,
        availability: torch.Tensor,
    ) -> tuple:
        """Return (mu, logvar) for the full latent space."""
        return self._fuse(batch_mods, availability)

    def encode_veh(
        self,
        batch_mods: dict,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return z_veh (sampled) for use by downstream heads.
        Encoder weights should be frozen when calling this.
        """
        mu, logvar = self._fuse(batch_mods, availability)
        mu_veh = mu[:, :self.z_veh_dim]
        logvar_veh = logvar[:, :self.z_veh_dim]
        return self.reparameterize(mu_veh, logvar_veh)

    # ------------------------------------------------------------------
    # Reparameterization
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, z_veh: torch.Tensor, availability: torch.Tensor) -> dict:
        """
        Decode z_veh for all available modalities.

        Returns {mod: reconstructed Tensor [B, C, T]} for present modalities.
        """
        recons = {}
        for i, mod in enumerate(MODALITIES):
            if availability[:, i].any():
                recons[mod] = self.decoders[mod](z_veh)
        return recons

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
          x_recon_t   : {mod: Tensor} — reconstructions of t-frame
          mu_t        : [B, z_total]
          logvar_t    : [B, z_total]
          z_t         : [B, z_total]  (reparameterised)
          z_env_next  : [B, z_env_dim]  (mean only, for slow loss)
          prior_mu_env    : [B, z_env_dim]
          prior_logvar_env: [B, z_env_dim]
        """
        # Encode t
        mu_t, logvar_t = self._fuse(batch_t, availability)
        z_t = self.reparameterize(mu_t, logvar_t)

        # Decode from z_veh only — z_env is excluded from reconstruction
        # so the decoder cannot use it as a shortcut.
        x_recon_t = self.decode(z_t[:, :self.z_veh_dim], availability)

        # Encode t+1 (mean only — used for slow loss on z_env)
        mu_next, _ = self._fuse(batch_next, availability)
        z_env_next = mu_next[:, self.z_veh_dim:]

        # iVAE prior for z_env conditioned on sensor domain
        prior_mu_env     = self.env_prior_mu(domain_ids)
        prior_logvar_env = self.env_prior_logvar(domain_ids)

        return {
            "x_recon_t":        x_recon_t,
            "mu_t":             mu_t,
            "logvar_t":         logvar_t,
            "z_t":              z_t,
            "z_env_next":       z_env_next,
            "prior_mu_env":     prior_mu_env,
            "prior_logvar_env": prior_logvar_env,
        }
