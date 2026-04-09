"""
LearnableFilterbank

Sinc-windowed bandpass FIR filters with learnable centre frequencies.
One instance per modality — audio and seismic are filtered independently
with separate weights and frequency ranges.

Forward pass:
    Input : (B, C, W)        — raw waveform batch
    Output: (B, K*C, T')     — log-compressed band-energy envelopes

where K = n_filters, T' = W // envelope_stride.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.config import ModalityConfig


# ---------------------------------------------------------------------------
# Sinc bandpass kernel initialisation
# ---------------------------------------------------------------------------

def sinc_bandpass(f_centers: torch.Tensor, bandwidth: torch.Tensor,
                  L: int, sr: int) -> torch.Tensor:
    """
    Build (K, 1, L) sinc-windowed bandpass FIR kernels.

    h_k(n) = 2*f_hi*sinc(2*f_hi*(n - L/2)) - 2*f_lo*sinc(2*f_lo*(n - L/2))
    windowed by Hann to reduce sidelobes.

    Args:
        f_centers : (K,) centre frequencies in Hz
        bandwidth : (K,) full bandwidth in Hz (f_hi - f_lo = bandwidth)
        L         : number of FIR taps
        sr        : sample rate in Hz

    Returns:
        kernels: (K, 1, L)
    """
    f_lo = (f_centers - bandwidth / 2).clamp(min=1.0)
    f_hi = (f_centers + bandwidth / 2).clamp(max=sr / 2 - 1.0)

    # Normalised frequencies (cycles per sample)
    f_lo_n = f_lo / sr
    f_hi_n = f_hi / sr

    # Time axis centred at L/2
    n = torch.arange(L, dtype=torch.float32, device=f_centers.device)
    t = n - L / 2 + 1e-8   # (L,) — offset avoids division by zero at centre

    # sinc(x) = sin(pi*x) / (pi*x)
    # h_bp(n) = 2*f_hi*sinc(2*f_hi*t) - 2*f_lo*sinc(2*f_lo*t)
    f_lo_n = f_lo_n.unsqueeze(1)   # (K, 1)
    f_hi_n = f_hi_n.unsqueeze(1)   # (K, 1)
    t = t.unsqueeze(0)              # (1, L)

    h = (2 * f_hi_n * torch.sinc(2 * f_hi_n * t)
         - 2 * f_lo_n * torch.sinc(2 * f_lo_n * t))   # (K, L)

    # Hann window
    hann = torch.hann_window(L, periodic=False, device=f_centers.device)   # (L,)
    h = h * hann.unsqueeze(0)

    # Normalise each kernel to unit L2 norm
    h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    return h.unsqueeze(1)   # (K, 1, L)


def _log_spaced_centers(f_min: float, f_max: float, K: int) -> torch.Tensor:
    """K centre frequencies log-spaced between f_min and f_max (Hz)."""
    return torch.exp(
        torch.linspace(math.log(f_min), math.log(f_max), K)
    )


def _vehicle_seismic_centers(K: int) -> torch.Tensor:
    """
    Seismic vehicle bands (ground-borne vibration):
        [2-8 Hz]    infrasound / distant rumble
        [8-20 Hz]   body resonance
        [20-40 Hz]  wheel-road interaction
        [40-70 Hz]  engine fundamental
        [70-100 Hz] upper harmonics
    Log-spaced within each band; centres rescaled to K filters total.
    """
    bands = [
        (2.0, 8.0),
        (8.0, 20.0),
        (20.0, 40.0),
        (40.0, 70.0),
        (70.0, 98.0),
    ]
    per_band = max(1, K // len(bands))
    centers = []
    for lo, hi in bands:
        centers.append(torch.exp(
            torch.linspace(math.log(lo), math.log(hi), per_band)
        ))
    c = torch.cat(centers)
    # Trim or pad to exactly K
    if len(c) > K:
        c = c[:K]
    elif len(c) < K:
        extras = torch.exp(
            torch.linspace(math.log(2.0), math.log(98.0), K - len(c))
        )
        c = torch.cat([c, extras])
    return c.sort().values


def _vehicle_audio_centers(K: int) -> torch.Tensor:
    """
    Acoustic vehicle bands (airborne sound) covering 20–7500 Hz.
    At 16 kHz the Nyquist is 8 kHz; 7500 Hz preserves engine harmonics
    through the 2–8 kHz region clipped by the former 4 kHz / 1800 Hz config.
        [20-150 Hz]    engine idle / exhaust fundamentals
        [150-600 Hz]   engine harmonics tier 1
        [600-2000 Hz]  tyre/road interaction
        [2000-5000 Hz] higher harmonics, intake noise
        [5000-7500 Hz] body panels, wind turbulence
    """
    bands = [
        (20.0,   150.0),
        (150.0,  600.0),
        (600.0,  2000.0),
        (2000.0, 5000.0),
        (5000.0, 7500.0),
    ]
    per_band = max(1, K // len(bands))
    centers = []
    for lo, hi in bands:
        centers.append(torch.exp(
            torch.linspace(math.log(lo), math.log(hi), per_band)
        ))
    c = torch.cat(centers)
    if len(c) > K:
        c = c[:K]
    elif len(c) < K:
        extras = torch.exp(
            torch.linspace(math.log(20.0), math.log(7500.0), K - len(c))
        )
        c = torch.cat([c, extras])
    return c.sort().values


# ---------------------------------------------------------------------------
# LearnableFilterbank
# ---------------------------------------------------------------------------

class LearnableFilterbank(nn.Module):
    """
    Depthwise sinc-windowed bandpass filterbank with learnable centre
    frequencies.  Instantiate one per modality.

    Args:
        mod_cfg : ModalityConfig for the target modality (sets sample_rate,
                  n_channels, n_filters, filter_len, freq_init, f_min, f_max,
                  envelope_pool, envelope_stride)
        sensor  : "seismic" or "audio" — selects vehicle-domain freq init
    """

    def __init__(self, mod_cfg: ModalityConfig, sensor: str = "seismic"):
        super().__init__()
        self.cfg = mod_cfg
        K = mod_cfg.n_filters
        C = mod_cfg.n_channels
        L = mod_cfg.filter_len
        sr = mod_cfg.sample_rate

        # Initialise centre frequencies
        if mod_cfg.freq_init == "vehicle":
            if sensor == "seismic":
                f_c = _vehicle_seismic_centers(K)
            else:
                f_c = _vehicle_audio_centers(K)
        else:  # "log"
            f_c = _log_spaced_centers(mod_cfg.f_min, mod_cfg.f_max, K)

        # Learnable log-centre-frequency parameters
        # Stored in log space so gradient descent can't push f below 0.
        self.log_f_center = nn.Parameter(torch.log(f_c))   # (K,)

        # Fixed bandwidth = f_center / 4 at init; could also be learnable.
        # For now, keep bandwidth proportional to centre (constant Q filter).
        self.register_buffer("_sr", torch.tensor(float(sr)))
        self._L = L
        self._K = K
        self._C = C

        # Envelope extraction: squared magnitude pooling
        self.pool = nn.AvgPool1d(
            kernel_size=mod_cfg.envelope_pool,
            stride=mod_cfg.envelope_stride,
        )

    def _build_kernels(self) -> torch.Tensor:
        """Reconstruct (K*C, 1, L) depthwise conv kernels from current params."""
        sr = self._sr.item()
        f_c = torch.exp(self.log_f_center).clamp(1.0, sr / 2 - 1.0)
        bw = (f_c / 4.0).clamp(min=1.0)
        kernels = sinc_bandpass(f_c, bw, self._L, sr)   # (K, 1, L)
        # Repeat C times for depthwise: (K*C, 1, L)
        return kernels.repeat(self._C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, W)
        Returns: (B, K*C, T')
        """
        _, C, _ = x.shape
        kernels = self._build_kernels()   # (K*C, 1, L)

        # Depthwise grouped convolution: groups=C means each input channel
        # is convolved with its own K kernels → output (B, K*C, W)
        pad = self._L // 2
        y = F.conv1d(x, kernels, padding=pad, groups=C)   # (B, K*C, W)

        y = y.pow(2)       # instantaneous power (B, K*C, W)
        y = self.pool(y)   # (B, K*C, T')
        y = torch.log1p(y) # log-compress: stabilises training
        return y

    @torch.no_grad()
    def center_frequencies(self) -> torch.Tensor:
        """Return current learnable centre frequencies in Hz (K,)."""
        return torch.exp(self.log_f_center).clamp(
            1.0, self._sr.item() / 2 - 1.0
        )
