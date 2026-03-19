import torch
import torchaudio


# ============================================================
# Mel Spectrogram Cache
# ============================================================

_mel_transforms = {}


def get_mel_transform(device, config):
    """Retrieve (or create and cache) a MelSpectrogram transform."""
    key = (device, config.REF_SAMPLE_RATE, config.N_FFT, config.HOP_LENGTH, config.MEL_BINS)

    if key not in _mel_transforms:
        _mel_transforms[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.REF_SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.MEL_BINS,
        ).to(device)

    return _mel_transforms[key]


def extract_mel_spectrogram(batch_tensor, config):
    """
    Convert 1D waveforms to 2D mel spectrograms.

    Input:  [B, C, T]
    Output: [B, C, MEL_BINS, FRAMES]
    """
    device = batch_tensor.device
    mel_transform = get_mel_transform(device, config)

    B, C, T = batch_tensor.shape
    flat_mels = mel_transform(batch_tensor.reshape(B * C, T))   # [B*C, MEL_BINS, FRAMES]
    return flat_mels.view(B, C, config.MEL_BINS, -1)


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

_EPSILON = 1e-8


def preprocess(batch_tensor, config):
    """
    Self-contained preprocessing pipeline for training and inference.

    Each window is independently normalised to zero mean and unit variance.
    No external calibration data (sigma, meta.pt) is required — the same
    function works identically at training time and at live inference.

    Steps:
      1. Per-window DC offset removal  (subtract mean along time axis)
      2. Per-window amplitude scaling   (divide by std along time axis)
      3. Transient spike clipping       (clamp to ±10)
      4. Optional mel spectrogram       (for 2D models)
    """
    # 1. DC offset removal — zero-mean each [C, T] window independently
    batch_tensor = batch_tensor - batch_tensor.mean(dim=-1, keepdim=True)

    # 2. Per-window z-score — unit variance per channel per window
    window_std = batch_tensor.std(dim=-1, keepdim=True)
    batch_tensor = batch_tensor / (window_std + _EPSILON)

    # 3. Spike clipping
    batch_tensor = batch_tensor.clamp(-10.0, 10.0)

    # 4. Frequency-domain transform
    if getattr(config, "USE_MEL", True):
        batch_tensor = extract_mel_spectrogram(batch_tensor, config)

    return batch_tensor
