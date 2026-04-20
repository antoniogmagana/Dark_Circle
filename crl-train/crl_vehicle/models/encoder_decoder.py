from __future__ import annotations
import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Transformer-based temporal encoder.
    Runs in fp32 to prevent softmax NaN on long sequences."""

    def __init__(
        self,
        in_channels: int,
        d_z: int = 24,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mu_head  = nn.Linear(d_model, d_z)
        self.lv_head  = nn.Linear(d_model, d_z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, T) → permute to (B, T, C) for transformer
        x = x.float().permute(0, 2, 1)
        h = self.norm_in(self.input_proj(x))
        h = self.transformer(h)
        h = h.mean(dim=1)   # (B, d_model)
        mu     = self.mu_head(h).clamp(-10.0, 10.0)
        logvar = self.lv_head(h).clamp(-4.0, 4.0)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar


class FeatureDecoder(nn.Module):
    """MLP decoder: z → (B, out_channels, seq_len)."""

    def __init__(
        self,
        out_channels: int,
        seq_len: int,
        d_z: int = 24,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels * seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(z.shape[0], self.out_channels, self.seq_len)
