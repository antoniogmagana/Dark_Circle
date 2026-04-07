"""
CausalEncoder

Maps SSM output (B, T', d_model) to a structured latent z with four
semantically-defined blocks:

    z_presence  (d_z_presence=1)  — is a vehicle present?
    z_type      (d_z_type=4)      — which vehicle category?
    z_proximity (d_z_proximity=1) — how close is it?
    z_noise     (d_z_noise=4)     — unstructured nuisance / sensor noise

Total d_z = 10.  Instantiate one per modality — each modality encodes
the same causal variables independently.

Reparameterisation: VAE-style (mu, log_var) → z.
split_z()  decomposes z into the four semantic blocks.
"""

import torch
import torch.nn as nn


class CausalEncoder(nn.Module):
    """
    Args:
        d_model      : SSM output dimension (= CRLConfig.d_model)
        d_z_presence : latent dim for vehicle presence (default 1)
        d_z_type     : latent dim for vehicle type (default 4)
        d_z_proximity: latent dim for proximity (default 1)
        d_z_noise    : latent dim for nuisance factors (default 4)
    """

    def __init__(
        self,
        d_model: int,
        d_z_presence: int = 1,
        d_z_type: int = 4,
        d_z_proximity: int = 1,
        d_z_noise: int = 4,
    ):
        super().__init__()
        self.d_z_presence = d_z_presence
        self.d_z_type = d_z_type
        self.d_z_proximity = d_z_proximity
        self.d_z_noise = d_z_noise
        self.d_z = d_z_presence + d_z_type + d_z_proximity + d_z_noise

        # Soft attention pooling: collapse T' timesteps → single context vector
        # Linear(d_model → 1) gives a scalar score per timestep; softmax over T'.
        self.attn_score = nn.Linear(d_model, 1)

        # Project context vector to (mu, log_var) for reparameterisation
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * self.d_z),
        )

        # Semantic index slices for split_z()
        s0 = 0
        s1 = s0 + d_z_presence
        s2 = s1 + d_z_type
        s3 = s2 + d_z_proximity
        s4 = s3 + d_z_noise
        self.presence_idx = slice(s0, s1)
        self.type_idx = slice(s1, s2)
        self.proximity_idx = slice(s2, s3)
        self.noise_idx = slice(s3, s4)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, T', d_model) — SSM output

        Returns:
            z       : (B, d_z)   reparameterised sample
            mu      : (B, d_z)   posterior mean
            log_var : (B, d_z)   posterior log-variance
        """
        # Attention pooling over T' steps
        scores = self.attn_score(x)           # (B, T', 1)
        attn = torch.softmax(scores, dim=1)   # (B, T', 1)
        ctx = (attn * x).sum(dim=1)           # (B, d_model)

        out = self.proj(ctx)                  # (B, 2*d_z)
        mu, log_var = out.chunk(2, dim=-1)    # each (B, d_z)

        # Clamp log_var to prevent KL explosion / posterior collapse
        log_var = log_var.clamp(-6.0, 4.0)

        # Reparameterise
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std                    # (B, d_z)

        return z, mu, log_var

    def split_z(self, z: torch.Tensor) -> tuple:
        """
        Decompose z into its four semantic blocks.

        Returns (z_presence, z_type, z_proximity, z_noise) where:
            z_presence  : (B, 1)  — sigmoid → [0, 1]
            z_type      : (B, 4)  — softmax → probability simplex
            z_proximity : (B, 1)  — sigmoid → [0, 1]
            z_noise     : (B, 4)  — unconstrained
        """
        z_presence = torch.sigmoid(z[:, self.presence_idx])
        z_type = torch.softmax(z[:, self.type_idx], dim=-1)
        z_proximity = torch.sigmoid(z[:, self.proximity_idx])
        z_noise = z[:, self.noise_idx]
        return z_presence, z_type, z_proximity, z_noise

    def split_z_raw(self, z: torch.Tensor) -> tuple:
        """
        Like split_z() but returns raw logits (no sigmoid/softmax).
        Use this when passing to a downstream linear head to avoid
        double-squashing.
        """
        z_presence = z[:, self.presence_idx]
        z_type = z[:, self.type_idx]
        z_proximity = z[:, self.proximity_idx]
        z_noise = z[:, self.noise_idx]
        return z_presence, z_type, z_proximity, z_noise
