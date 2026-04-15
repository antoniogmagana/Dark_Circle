import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, d_z=10, d_model=64, n_heads=4, n_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.fc_mu = nn.Linear(d_model, d_z)
        self.fc_logvar = nn.Linear(d_model, d_z)

    def forward(self, x):
        # x: (B, C, T) from frontend
        x = x.permute(0, 2, 1)          # (B, T, C)
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.transformer(x)          # (B, T, d_model)
        x = x.mean(dim=1)               # (B, d_model)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-4, 4)

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu

        return z, mu, logvar


class FeatureDecoder(nn.Module):
    def __init__(self, out_channels, seq_len, d_z=10, d_model=64):
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len

        self.mlp = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels * seq_len),
        )

    def forward(self, z):
        # z: (B, d_z)
        x = self.mlp(z)                                                # (B, out_channels * seq_len)
        return x.reshape(x.shape[0], self.out_channels, self.seq_len)  # (B, C, T')
