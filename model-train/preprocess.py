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


def extract_mel_spectrogram(
    signal_tensor,
    n_mels=config.MEL_BINS,
    hop_length=config.MEL_HOP_LENGTH,
    top_db=config.MEL_TOP_DB,
):
    mel_transform = T.MelSpectrogram(
        sample_rate=config.ACOUSTIC_SR, n_mels=n_mels, hop_length=hop_length
    ).to(config.DEVICE)

    amplitude_to_db = T.AmplitudeToDB(top_db=top_db).to(config.DEVICE)

    mel_power = mel_transform(signal_tensor)
    return amplitude_to_db(mel_power)
