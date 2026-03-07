import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import config


def zero_center_window(data_tensor):
    local_means = torch.mean(data_tensor, dim=-1, keepdim=True)
    return data_tensor - local_means


def standardize_global(centered_data, global_std):
    if not isinstance(global_std, torch.Tensor):
        global_std = torch.tensor(global_std, device=config.DEVICE)
    safe_std = torch.where(
        global_std == 0, torch.tensor(1e-6, device=config.DEVICE), global_std
    )
    return centered_data / safe_std


def calculate_smv(accel_tensor):
    smv = torch.sqrt(torch.sum(accel_tensor**2, dim=1, keepdim=True))
    return zero_center_window(smv)


def align_and_upsample(signal_tensor, target_length=config.ACOUSTIC_SR):
    if signal_tensor.shape[-1] == target_length:
        return signal_tensor
    return F.interpolate(
        signal_tensor, size=target_length, mode="linear", align_corners=True
    )


def extract_mel_spectrogram(batch_tensor):
    """
    Converts wave batches to Spectrograms.
    Note: batch_tensor must already be on config.DEVICE (GPU).
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=config.ACOUSTIC_SR,
        n_mels=config.MEL_BINS,
        hop_length=config.MEL_HOP_LENGTH,
    ).to(batch_tensor.device)

    amplitude_to_db = T.AmplitudeToDB(top_db=config.MEL_TOP_DB).to(batch_tensor.device)

    # MelSpectrogram natively supports [Batch, Channel, Time]
    return amplitude_to_db(mel_transform(batch_tensor))


def parallel_batch_process(batch_tensor, target_sr=config.ACOUSTIC_SR):
    """
    Takes a 3D tensor [Batch, Channels, Time] and processes it using
    vectorized operations. This triggers Intel MKL/AVX-512 to use all 120 cores.
    """
    # 1. Zero Center (Vectorized across the whole batch)
    batch_tensor = batch_tensor - batch_tensor.mean(dim=-1, keepdim=True)

    # 2. Vectorized Upsampling
    # Interpolate expects 4D: [Batch, Channels, Depth, Width]
    # We treat 'Time' as Width and add a dummy Depth dimension of 1.
    if batch_tensor.shape[-1] != target_sr:
        batch_tensor = batch_tensor.unsqueeze(2)  # [B, C, 1, T_native]
        batch_tensor = F.interpolate(
            batch_tensor, size=(1, target_sr), mode="linear", align_corners=True
        ).squeeze(
            2
        )  # Back to [B, C, T_target]

    return batch_tensor
