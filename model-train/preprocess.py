import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

import config


# ============================================================
# 1. Zero-centering and upsampling
# ============================================================


def preprocess_window(window_tensor, target_sr=config.ACOUSTIC_SR):
    """
    Preprocess a single 1-second window.

    Input:
        window_tensor: [C, T_native]
            - C = number of channels (audio, seismic, accel axes)
            - T_native = native_sr * SAMPLE_SECONDS

    Output:
        [C, target_sr * SAMPLE_SECONDS]
            - typically [C, 16000]
    """

    # Zero-center each channel
    window_tensor = window_tensor - window_tensor.mean(dim=-1, keepdim=True)

    C, T_native = window_tensor.shape
    target_length = target_sr * config.SAMPLE_SECONDS  # usually 16000

    # Upsample if needed
    if T_native != target_length:
        window_tensor = F.interpolate(
            window_tensor.unsqueeze(0),  # [1, C, T_native]
            size=target_length,
            mode="linear",
            align_corners=True,
        ).squeeze(
            0
        )  # [C, target_length]

    return window_tensor


# ============================================================
# 2. Batch preprocessing
# ============================================================


def preprocess_batch(batch_tensor):
    """
    Preprocess a batch of 1-second windows.

    Input:
        batch_tensor: [B, C, T_native]

    Output:
        [B, C, target_sr]
    """
    B, C, T_native = batch_tensor.shape
    processed = []

    for i in range(B):
        processed.append(preprocess_window(batch_tensor[i]))

    return torch.stack(processed, dim=0)


# ============================================================
# 3. Mel spectrogram extraction
# ============================================================

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.ACOUSTIC_SR,
    n_fft=2048,
    hop_length=config.MEL_HOP_LENGTH,
    n_mels=config.MEL_BINS,
)


def extract_mel_spectrogram(batch_tensor):
    """
    Convert waveform batch into mel spectrograms.

    Input:
        batch_tensor: [B, C, T] after upsampling

    Output:
        [B, C, MEL_BINS, time_frames]
    """
    B, C, T = batch_tensor.shape
    mel_list = []

    for c in range(C):
        mel = _mel_transform(batch_tensor[:, c, :])  # [B, MEL_BINS, frames]
        mel_list.append(mel)

    return torch.stack(mel_list, dim=1)  # [B, C, MEL_BINS, frames]


# ============================================================
# 4. Convenience wrapper for training
# ============================================================


def preprocess_for_training(batch_tensor, use_mel=True):
    """
    Full preprocessing pipeline for training.

    Input:
        batch_tensor: [B, C, T_native]

    Output:
        If use_mel:
            [B, C, MEL_BINS, frames]
        Else:
            [B, C, target_sr]
    """
    batch_tensor = preprocess_batch(batch_tensor)

    if use_mel:
        batch_tensor = extract_mel_spectrogram(batch_tensor)

    return batch_tensor
