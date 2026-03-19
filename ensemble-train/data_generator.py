import torch


def generate_white_noise(config, window_length=None, num_channels=None, amplitude=0.01):
    """Generate white noise on CPU (workers move to GPU later)."""
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    return (torch.randn(num_channels, window_length) * amplitude).float()


def generate_no_vehicle_sample(
    config, window_length=None, num_channels=None,
    noise_profile="environmental", amplitude=None,
):
    """
    Generate a synthetic background sample.

    ``amplitude`` can be:
      - None         → use a small constant default
      - a float      → uniform across channels
      - a Tensor     → per-channel noise floor (from calibration)
    """
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    # Normalise amplitude to shape [C, 1]
    if amplitude is None:
        amp = torch.full((num_channels, 1), 0.01)
    elif isinstance(amplitude, torch.Tensor):
        amp = amplitude.view(num_channels, 1).cpu()
    else:
        amp = torch.tensor(amplitude).view(1, 1)

    base_noise = torch.randn(num_channels, window_length)

    if noise_profile == "sensor_hiss":
        return (base_noise * amp).float()

    # "environmental" — low-pass filtered rumble
    kernel_size = 51
    kernel = torch.ones(1, 1, kernel_size) / kernel_size
    noise_3d = base_noise.view(num_channels, 1, window_length)

    rumble = torch.nn.functional.conv1d(noise_3d, kernel, padding=kernel_size // 2)
    max_vals = rumble.abs().amax(dim=2, keepdim=True) + 1e-8
    rumble = rumble / max_vals

    return (rumble.view(num_channels, window_length) * amp).float()


def inject_snr_noise(clean_signal, target_snr_db):
    """Add noise at a specified SNR (dB) based on AC signal power."""
    ac_signal = clean_signal - clean_signal.mean()
    signal_power = (ac_signal ** 2).mean()

    if signal_power == 0:
        return clean_signal

    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = torch.randn_like(clean_signal) * noise_power.sqrt()
    return (clean_signal + noise).float()


def augment_batch(batch_tensor, snr_range=(10, 30)):
    """Vectorised SNR augmentation for an entire batch."""
    B = batch_tensor.shape[0]
    device = batch_tensor.device

    random_snrs = torch.zeros(B, device=device).uniform_(snr_range[0], snr_range[1])

    ac = batch_tensor - batch_tensor.mean(dim=-1, keepdim=True)
    signal_powers = (ac ** 2).mean(dim=(1, 2), keepdim=True)
    noise_powers = signal_powers / (10 ** (random_snrs.view(-1, 1, 1) / 10))

    noise = torch.randn_like(batch_tensor) * noise_powers.sqrt()
    return (batch_tensor + noise).float()
