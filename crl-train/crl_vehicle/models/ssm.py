"""
TemporalSSM

Models temporal dynamics of filterbank band-energy envelopes.
Input : (B, K*C, T')   — filterbank output, channels-first
Output: (B, T', d_model) — contextualised per-timestep representations

Default implementation: Causal Transformer (Option C from the plan).
A Mamba-based alternative is available when mamba-ssm is installed
(set backend="mamba" at construction time).

Shape contract:
    The filterbank output has K*C channels and T' time steps.
    A learned linear projection maps K*C → d_model before the SSM.
    Output keeps the (B, T', d_model) layout for the CausalEncoder.
"""

import torch
import torch.nn as nn

from crl_vehicle.config import CRLConfig


# ---------------------------------------------------------------------------
# Causal Transformer backend
# ---------------------------------------------------------------------------

class _CausalTransformerSSM(nn.Module):
    """
    Standard Transformer encoder with a lower-triangular causal mask so
    each timestep can only attend to itself and past timesteps.
    Works well for T' <= 512 (T'=25 in this pipeline).
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # expects (B, T, d_model)
            norm_first=True,    # Pre-LN: more stable gradients
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T', d_model) → (B, T', d_model)"""
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )
        return self.encoder(x, mask=mask, is_causal=True)


# ---------------------------------------------------------------------------
# Mamba backend (optional)
# ---------------------------------------------------------------------------

def _try_mamba_backend(d_model: int, d_state: int, d_conv: int, expand: int):
    """
    Attempt to import and wrap Mamba. Returns None if mamba-ssm is not
    installed so the caller can fall back to the Transformer backend.

    Install with:  pip install mamba-ssm causal-conv1d
    """
    try:
        from mamba_ssm import Mamba

        class _MambaSSM(nn.Module):
            def __init__(self):
                super().__init__()
                self.ssm = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, T', d_model) — Mamba expects (B, L, D)
                return self.ssm(x)

        return _MambaSSM()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# TemporalSSM  (public interface)
# ---------------------------------------------------------------------------

class TemporalSSM(nn.Module):
    """
    Projects filterbank output K*C → d_model, then runs a temporal model.

    Args:
        in_channels : K*C (filterbank_out_channels from ModalityConfig)
        config      : CRLConfig (provides d_model, ssm_nhead, ssm_layers,
                      ssm_dropout)
        backend     : "transformer" (default) | "mamba"
        mamba_d_state, mamba_d_conv, mamba_expand : Mamba hyperparams
    """

    def __init__(
        self,
        in_channels: int,
        config: CRLConfig,
        backend: str = "transformer",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.d_model = config.d_model

        # Project filterbank output to d_model
        self.proj_in = nn.Sequential(
            nn.Linear(in_channels, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Positional encoding (learned, since T'=25 is very short)
        # Max sequence length: use 256 as a safe upper bound.
        self.pos_embed = nn.Embedding(256, config.d_model)

        # Select backend
        if backend == "mamba":
            ssm = _try_mamba_backend(
                config.d_model, mamba_d_state, mamba_d_conv, mamba_expand
            )
            if ssm is None:
                print(
                    "  [TemporalSSM] mamba-ssm not found; "
                    "falling back to Causal Transformer."
                )
                ssm = _CausalTransformerSSM(
                    config.d_model,
                    config.ssm_nhead,
                    config.ssm_layers,
                    config.ssm_dropout,
                )
        else:
            ssm = _CausalTransformerSSM(
                config.d_model,
                config.ssm_nhead,
                config.ssm_layers,
                config.ssm_dropout,
            )

        self.ssm = ssm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, K*C, T')  — filterbank output, channels-first
        Returns: (B, T', d_model)
        """
        B, KC, T = x.shape
        x = x.permute(0, 2, 1)           # (B, T', K*C)
        x = self.proj_in(x)              # (B, T', d_model)

        pos = torch.arange(T, device=x.device)
        x = x + self.pos_embed(pos)      # (B, T', d_model)

        x = self.ssm(x)                  # (B, T', d_model)
        return x
