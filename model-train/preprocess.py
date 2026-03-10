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


def preprocess_for_training(batch_tensor, mu, sigma, epsilon, use_mel=True):
    """
    Full preprocessing pipeline for training and evaluation.
    """
    # 1. Standardize using our Train set stats
    # Ensure mu and sigma are on the same device as batch_tensor
    mu = mu.to(batch_tensor.device)
    sigma = sigma.to(batch_tensor.device)

    # Reshape from [Channel] to [1, Channel, 1] for broadcasting
    mu = mu.view(1, -1, 1)
    sigma = sigma.view(1, -1, 1)

    # Now this will execute perfectly
    batch_tensor = (batch_tensor - mu) / (sigma + epsilon)
    # Normalize each window
    window_mean = batch_tensor.mean(dim=-1, keepdim=True)
    window_std = batch_tensor.std(dim=-1, keepdim=True)
    batch_tensor = (batch_tensor - window_mean) / (window_std + epsilon)

    # 2. Convert to frequency domain
    if use_mel:
        batch_tensor = extract_mel_spectrogram(batch_tensor)

    return batch_tensor
