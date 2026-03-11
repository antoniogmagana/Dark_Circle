import torch
import torchaudio
import config



# ============================================================
# 1. Vectorized Mel Spectrogram Extraction
# ============================================================

_mel_transform = None


def get_mel_transform(device):
    global _mel_transform
    if (
        _mel_transform is None
        or next(_mel_transform.parameters(), torch.tensor([], device=device)).device
        != device
    ):
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.REF_SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.MEL_BINS,
        ).to(device)
    return _mel_transform


def extract_mel_spectrogram(batch_tensor):
    """
    Converts 1D waveforms to 2D Mel Spectrograms.
    Optimized to process all channels and batches simultaneously without for-loops.

    Input: [B, C, T] -> Output: [B, C, MEL_BINS, FRAMES]
    """
    device = batch_tensor.device
    mel_transform = get_mel_transform(device)

    B, C, T = batch_tensor.shape

    # Flatten the Batch and Channel dimensions together for massive GPU speedup
    # Shape becomes: [B * C, T]
    flat_batch = batch_tensor.reshape(B * C, T)

    # Process everything in one single pass
    flat_mels = mel_transform(flat_batch)  # Shape: [B * C, MEL_BINS, FRAMES]

    # Reshape back to the expected 4D tensor
    mels = flat_mels.view(B, C, config.MEL_BINS, -1)

    return mels


# ============================================================
# 2. Main Training Wrapper
# ============================================================


def preprocess_for_training(batch_tensor, sigma, epsilon, use_mel=True):
    """
    Robust preprocessing pipeline for deployment environments.
    Handles DC drift, preserves physical amplitude, and limits transient spikes.
    """
    # 1. LOCAL DC OFFSET REMOVAL
    window_mean = batch_tensor.mean(dim=-1, keepdim=True)
    batch_tensor = batch_tensor - window_mean

    # 2. GLOBAL AMPLITUDE SCALING
    sigma = sigma.to(batch_tensor.device).view(1, -1, 1)
    batch_tensor = batch_tensor / (sigma + epsilon)

    # 3. TRANSIENT SPIKE CLIPPING
    batch_tensor = torch.clamp(batch_tensor, min=-10.0, max=10.0)

    # 4. Convert to frequency domain
    if use_mel:
        batch_tensor = extract_mel_spectrogram(batch_tensor)

    return batch_tensor