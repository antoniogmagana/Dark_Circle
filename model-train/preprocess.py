import torch
import torchaudio
import config

# ============================================================
# 1. True Standardization (Z-Score)
# ============================================================


def standardize_batch(batch_tensor, channel_maxs, eps=1e-8):
    """
    batch_tensor: [B, C, T]
    channel_maxs: [C] (The maximum amplitude of each channel from the training set)
    """
    # 1. Window-level Mean Subtraction (Fixes sensor hardware drift/DC offset)
    mean = batch_tensor.mean(dim=-1, keepdim=True)
    centered = batch_tensor - mean

    # 2. Global-level Scaling (Preserves relative volume, prevents gradient explosion)
    # We reshape channel_maxs from [C] to [1, C, 1] so it broadcasts across batches and time
    scaled = centered / (channel_maxs.view(1, -1, 1) + eps)

    return scaled


# ============================================================
# 2. Vectorized Mel Spectrogram Extraction
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
    flat_batch = batch_tensor.view(B * C, T)

    # Process everything in one single pass
    flat_mels = mel_transform(flat_batch)  # Shape: [B * C, MEL_BINS, FRAMES]

    # Reshape back to the expected 4D tensor
    mels = flat_mels.view(B, C, config.MEL_BINS, -1)

    return mels


# ============================================================
# 3. Main Training Wrapper
# ============================================================


def preprocess_for_training(batch_tensor, channel_maxs, use_mel=True):
    """
    Full preprocessing pipeline for training and evaluation.
    """
    # 1. Standardize using our Train set stats
    batch_tensor = standardize_batch(batch_tensor, channel_maxs)

    # 2. Convert to frequency domain
    if use_mel:
        batch_tensor = extract_mel_spectrogram(batch_tensor)

    return batch_tensor
