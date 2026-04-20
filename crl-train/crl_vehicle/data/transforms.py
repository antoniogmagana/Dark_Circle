"""Signal transforms: DC removal and 7 intervention noise types."""
from __future__ import annotations
import math
import torch

N_INTERVENTIONS = 7


def remove_dc(x: torch.Tensor) -> torch.Tensor:
    """Per-channel mean subtraction (DC removal). Preserves relative amplitudes. x: (C, W)."""
    return x - x.mean(dim=-1, keepdim=True)


def _white_noise(n: int, sr: int) -> torch.Tensor:
    return torch.randn(n)


def _brown_noise(n: int, sr: int) -> torch.Tensor:
    white = torch.randn(n)
    return torch.cumsum(white, dim=0) / math.sqrt(n)


def _pink_noise(n: int, sr: int) -> torch.Tensor:
    f = torch.fft.rfftfreq(n).clamp(min=1e-6)
    spectrum = torch.fft.rfft(torch.randn(n)) / f.sqrt()
    return torch.fft.irfft(spectrum, n=n)


def _green_noise(n: int, sr: int) -> torch.Tensor:
    f = torch.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = torch.fft.rfft(torch.randn(n))
    mask = ((f >= 300) & (f <= 800)).float()
    return torch.fft.irfft(spectrum * mask, n=n)


def _low_freq_osc(n: int, sr: int) -> torch.Tensor:
    t = torch.linspace(0, n / sr, n)
    freq = 2.0 + torch.rand(1).item() * 10.0
    return torch.sin(2 * math.pi * freq * t)


def _high_freq_chirp(n: int, sr: int) -> torch.Tensor:
    t = torch.linspace(0, n / sr, n)
    f0, f1 = 1000.0, min(sr / 2 - 1, 8000.0)
    k = (f1 - f0) / (n / sr)
    return torch.cos(2 * math.pi * (f0 * t + 0.5 * k * t ** 2))


def _bird_chirps(n: int, sr: int) -> torch.Tensor:
    noise = torch.randn(n)
    f_c = 2000.0 + torch.rand(1).item() * 2000.0
    f = torch.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = torch.fft.rfft(noise)
    bw = 500.0
    mask = torch.exp(-0.5 * ((f - f_c) / bw) ** 2)
    return torch.fft.irfft(spectrum * mask, n=n)


_GENERATORS = [
    _white_noise,
    _brown_noise,
    _pink_noise,
    _green_noise,
    _low_freq_osc,
    _high_freq_chirp,
    _bird_chirps,
]


def apply_intervention(
    x: torch.Tensor, intervention_id: int, sample_rate: int
) -> torch.Tensor:
    """Add noise type `intervention_id` (1-7) at 20% RMS of signal. x: (C, W)."""
    C, W = x.shape
    gen = _GENERATORS[intervention_id - 1]
    noise = gen(W, sample_rate)
    signal_rms = x.pow(2).mean().sqrt().clamp(min=1e-8)
    noise_rms  = noise.pow(2).mean().sqrt().clamp(min=1e-8)
    noise = noise * (0.2 * signal_rms / noise_rms)
    return x + noise.unsqueeze(0).expand(C, -1)
