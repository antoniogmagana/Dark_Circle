import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalVAE(nn.Module):
    def __init__(self, num_sensors, z_veh_dim=16, z_env_dim=16):
        super().__init__()
        self.z_veh_dim = z_veh_dim
        self.z_env_dim = z_env_dim
        self.z_total = z_veh_dim + z_env_dim

        # 1. ENCODER (CNN to process spectrograms)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 32, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 16, 8)
            nn.ReLU(),
            nn.Flatten(),
        )
        # Assuming input is (1, 64, 32), flatten size is 32 * 16 * 8 = 4096
        self.fc_mu = nn.Linear(4096, self.z_total)
        self.fc_logvar = nn.Linear(4096, self.z_total)

        # 2. iVAE CONDITIONAL PRIOR FOR Z_ENV
        # Maps the sensor ID (u) to a specific prior mean/variance for the environment noise
        self.env_prior_mu = nn.Embedding(num_sensors, z_env_dim)
        self.env_prior_logvar = nn.Embedding(num_sensors, z_env_dim)

        # 3. DECODER
        self.fc_dec = nn.Linear(self.z_total, 4096)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 32, 16, 8)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x_t, x_next, u):
        # Encode time T
        mu_t, logvar_t = self.encode(x_t)
        z_t = self.reparameterize(mu_t, logvar_t)

        # Encode time T+1 (for slow loss)
        mu_next, _ = self.encode(x_next)
        z_next = mu_next  # Only need the mean for the slow penalty

        # Decode time T
        x_recon = self.decode(z_t)

        return x_recon, mu_t, logvar_t, z_t, z_next
