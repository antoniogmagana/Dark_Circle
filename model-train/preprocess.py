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
    # batch_tensor: [B, C, T]
    device = batch_tensor.device
    mel_transform = get_mel_transform(device)

    B, C, T = batch_tensor.shape
    mels = []
    for c in range(C):
        mel = mel_transform(batch_tensor[:, c, :])  # [B, MEL_BINS, frames]
        mels.append(mel)
    return torch.stack(mels, dim=1)  # [B, C, MEL_BINS, frames]


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


def resample_to(x: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    x: [C, T_orig]
    returns: [C, T_target]
    """
    if orig_sr == target_sr:
        return x
    C, T = x.shape
    new_T = int(round(T * target_sr / orig_sr))
    x = x.unsqueeze(0)  # [1, C, T]
    x = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
    return x.squeeze(0)
