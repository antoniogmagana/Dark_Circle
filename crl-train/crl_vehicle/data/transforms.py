"""Signal transforms: DC removal and 7 intervention noise types.

The intervention generators come in two API shapes:

* per-sample (`apply_intervention`) — takes (C, W) and returns (C, W). Kept for
  back-compat with any caller that still wants the old behaviour.
* batched (`apply_intervention_batch`) — takes (B, C, W) and returns (B, C, W),
  vectorized on whatever device the input lives on. Use this in training code:
  per-sample interventions on CPU FFTs were the dominant bottleneck.
"""

from __future__ import annotations

import math

import torch

N_INTERVENTIONS = 7


def remove_dc(x: torch.Tensor) -> torch.Tensor:
    """Per-channel mean subtraction (DC removal). Last dim is time."""
    return x - x.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Per-sample generators (legacy API). Kept for back-compat — used by tests
# and by the dataset's worker fallback path. New training code should call
# the batched generators below.
# ---------------------------------------------------------------------------


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
    return torch.cos(2 * math.pi * (f0 * t + 0.5 * k * t**2))


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


def apply_intervention(x: torch.Tensor, intervention_id: int, sample_rate: int) -> torch.Tensor:
    """Add noise type `intervention_id` (1-7) at 20% RMS of signal. x: (C, W)."""
    C, W = x.shape
    gen = _GENERATORS[intervention_id - 1]
    noise = gen(W, sample_rate)
    signal_rms = x.pow(2).mean().sqrt().clamp(min=1e-8)
    noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)
    noise = noise * (0.2 * signal_rms / noise_rms)
    return x + noise.unsqueeze(0).expand(C, -1)


# ---------------------------------------------------------------------------
# Batched generators. Each takes (B, W) and returns (B, W) on the input
# device. These are the ones the training loop should call.
# ---------------------------------------------------------------------------


def _white_noise_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    return torch.randn(B, W, device=device, dtype=dtype)


def _brown_noise_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    white = torch.randn(B, W, device=device, dtype=dtype)
    return torch.cumsum(white, dim=1) / math.sqrt(W)


def _pink_noise_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    f = torch.fft.rfftfreq(W, device=device).clamp(min=1e-6)
    spectrum = torch.fft.rfft(torch.randn(B, W, device=device, dtype=dtype), dim=1)
    return torch.fft.irfft(spectrum / f.sqrt(), n=W, dim=1)


def _green_noise_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    f = torch.fft.rfftfreq(W, d=1.0 / sr, device=device)
    spectrum = torch.fft.rfft(torch.randn(B, W, device=device, dtype=dtype), dim=1)
    mask = ((f >= 300) & (f <= 800)).to(dtype)
    return torch.fft.irfft(spectrum * mask, n=W, dim=1)


def _low_freq_osc_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    t = torch.linspace(0, W / sr, W, device=device, dtype=dtype)
    freq = 2.0 + torch.rand(B, 1, device=device, dtype=dtype) * 10.0
    return torch.sin(2 * math.pi * freq * t)


def _high_freq_chirp_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    t = torch.linspace(0, W / sr, W, device=device, dtype=dtype)
    f0 = 1000.0
    f1 = min(sr / 2 - 1, 8000.0)
    k = (f1 - f0) / (W / sr)
    chirp = torch.cos(2 * math.pi * (f0 * t + 0.5 * k * t**2))
    return chirp.unsqueeze(0).expand(B, W).contiguous()


def _bird_chirps_b(B: int, W: int, sr: int, device, dtype) -> torch.Tensor:
    noise = torch.randn(B, W, device=device, dtype=dtype)
    f_c = 2000.0 + torch.rand(B, 1, device=device, dtype=dtype) * 2000.0
    f = torch.fft.rfftfreq(W, d=1.0 / sr, device=device).unsqueeze(0)  # (1, F)
    spectrum = torch.fft.rfft(noise, dim=1)
    bw = 500.0
    mask = torch.exp(-0.5 * ((f - f_c) / bw) ** 2)
    return torch.fft.irfft(spectrum * mask, n=W, dim=1)


_BATCH_GENERATORS = [
    _white_noise_b,
    _brown_noise_b,
    _pink_noise_b,
    _green_noise_b,
    _low_freq_osc_b,
    _high_freq_chirp_b,
    _bird_chirps_b,
]


def apply_intervention_batch(
    x: torch.Tensor,
    sample_rate: int,
    interv_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a per-sample random intervention to a batch.

    Args:
        x: (B, C, W). Augmentation is the same across channels (channel-axis
            tile of a single (B, W) noise tensor), matching the legacy
            per-sample apply_intervention contract.
        sample_rate: Hz, used by frequency-domain generators.
        interv_ids: optional (B,) long tensor of intervention IDs in
            [0, N_INTERVENTIONS]. ID 0 = no-op (sample passes through).
            If None, sampled uniformly per batch row.

    Returns:
        (B, C, W) tensor on the same device/dtype as x.
    """
    B, C, W = x.shape
    device, dtype = x.device, x.dtype

    if interv_ids is None:
        interv_ids = torch.randint(0, N_INTERVENTIONS + 1, (B,), device=device)
    elif interv_ids.device != device:
        interv_ids = interv_ids.to(device)

    out = x.clone()
    sig_rms = x.pow(2).mean(dim=(1, 2)).sqrt().clamp(min=1e-8)  # (B,)

    # Each generator runs only on the rows that need it. Two wins over the
    # legacy per-id approach:
    #   1. We avoid host syncs (`mask.any()`) by iterating only the IDs that
    #      appear in `interv_ids`, computed once via `unique`.
    #   2. Each generator is given only the subset size it needs, not B,
    #      saving ~5–7× audio FFT work on a typical batch.
    # `interv_ids.unique()` returns a CPU tensor when called on CPU; on GPU it
    # forces one D→H sync per batch — far cheaper than 7 wasted generators.
    unique_ids = interv_ids.unique().tolist()
    for k in unique_ids:
        if k == 0 or k > len(_BATCH_GENERATORS):
            continue
        mask = interv_ids == k
        idx = mask.nonzero(as_tuple=True)[0]
        n_k = idx.numel()
        if n_k == 0:
            continue
        gen = _BATCH_GENERATORS[k - 1]
        noise = gen(n_k, W, sample_rate, device, dtype)  # (n_k, W)
        noise_rms = noise.pow(2).mean(dim=1).sqrt().clamp(min=1e-8)  # (n_k,)
        scale = (0.2 * sig_rms[idx] / noise_rms).unsqueeze(-1)  # (n_k, 1)
        scaled = (noise * scale).unsqueeze(1).expand(n_k, C, W)  # (n_k, C, W)
        out[idx] = out[idx] + scaled

    return out
