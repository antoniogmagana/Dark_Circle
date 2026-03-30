import torch

# NOTICE: global 'config' is no longer imported


def generate_white_noise(config, window_length=None, num_channels=None, amplitude=0.01):
    # Dynamic defaults synced with injected configuration
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    # CRITICAL FIX: Workers generate [C, T] on CPU. Main thread moves to GPU later.
    noise = torch.randn((num_channels, window_length), device="cpu") * amplitude
    return noise.to(torch.float32)


def generate_no_vehicle_sample(
    config, window_length=None, num_channels=None, noise_profile="environmental", amplitude=None
):
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    # Handle per-channel dynamic amplitude
    if amplitude is None:
        # Fallback if none is provided
        amplitude_tensor = torch.full((num_channels, 1), 0.01, device="cpu")
    elif isinstance(amplitude, torch.Tensor):
        # Reshape to [Channels, 1] so it multiplies across the time dimension correctly
        amplitude_tensor = amplitude.view(num_channels, 1).to("cpu")
    else:
        # Fallback for floats
        amplitude_tensor = torch.tensor(amplitude, device="cpu").view(1, 1)

    base_noise = torch.randn((num_channels, window_length), device="cpu")

    if noise_profile == "sensor_hiss":
        return (base_noise * amplitude_tensor).to(torch.float32)

    elif noise_profile == "environmental":
        kernel_size = 51
        kernel = torch.ones((1, 1, kernel_size), device="cpu") / kernel_size
        noise_reshaped = base_noise.view(num_channels, 1, window_length)

        environmental_rumble = torch.nn.functional.conv1d(
            noise_reshaped, kernel, padding=kernel_size // 2
        )

        max_vals = (
            torch.max(torch.abs(environmental_rumble), dim=2, keepdim=True)[0] + 1e-8
        )
        environmental_rumble = environmental_rumble / max_vals

        # Multiply by our dynamic, per-channel noise floor
        return (environmental_rumble.view(num_channels, window_length) * amplitude_tensor).to(
            torch.float32
        )


def inject_snr_noise(clean_signal, target_snr_db):
    """Injects noise based strictly on the AC power of the signal."""
    # Temporarily remove DC offset to calculate true AC signal power
    ac_signal = clean_signal - torch.mean(clean_signal)
    signal_power = torch.mean(ac_signal**2)
    
    if signal_power == 0:
        return clean_signal

    noise_power = signal_power / (10 ** (target_snr_db / 10))
    # randn_like automatically inherits the device of clean_signal
    noise = torch.randn_like(clean_signal) * torch.sqrt(noise_power)
    
    # Add the noise back to the ORIGINAL signal (preserving its DC offset for later)
    return (clean_signal + noise).to(torch.float32)


def augment_batch(batch_tensor, snr_range=(10, 30)):
    """Vectorized SNR augmentation based strictly on AC power."""
    batch_size = batch_tensor.shape[0]
    
    # Use the batch's dynamic device so it scales whether on CPU or GPU
    random_snrs = torch.zeros(batch_size, device=batch_tensor.device).uniform_(
        snr_range[0], snr_range[1]
    )

    # Temporarily remove DC offset per-window to calculate true AC signal power
    window_means = batch_tensor.mean(dim=-1, keepdim=True)
    ac_signal = batch_tensor - window_means
    
    signal_powers = torch.mean(ac_signal**2, dim=(1, 2), keepdim=True)
    noise_powers = signal_powers / (10 ** (random_snrs.view(-1, 1, 1) / 10))

    noise = torch.randn_like(batch_tensor) * torch.sqrt(noise_powers)
    
    # Add the noise back to the ORIGINAL batch (preserving DC offsets for later)
    return (batch_tensor + noise).to(torch.float32)