import torch
import config


def generate_white_noise(window_length=None, num_channels=None, amplitude=0.01):
    # Dynamic defaults synced with global configuration
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    # CRITICAL FIX: Workers generate [C, T] on CPU. Main thread moves to GPU later.
    noise = torch.randn((num_channels, window_length), device="cpu") * amplitude
    return noise.to(torch.float32)


def generate_no_vehicle_sample(
    window_length=None, num_channels=None, noise_profile="environmental", amplitude=0.01
):
    if window_length is None:
        window_length = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
    if num_channels is None:
        num_channels = config.IN_CHANNELS

    # CRITICAL FIX: Device set explicitly to CPU, Shape set to [C, T]
    base_noise = torch.randn((num_channels, window_length), device="cpu")

    if noise_profile == "sensor_hiss":
        return (base_noise * amplitude).to(torch.float32)

    elif noise_profile == "environmental":
        kernel_size = 51
        # Conv1d expects [Batch, Channels, Time]. We treat num_channels as the Batch dimension.
        kernel = torch.ones((1, 1, kernel_size), device="cpu") / kernel_size
        noise_reshaped = base_noise.view(num_channels, 1, window_length)

        environmental_rumble = torch.nn.functional.conv1d(
            noise_reshaped, kernel, padding=kernel_size // 2
        )

        # Safe normalization per-channel to avoid divide-by-zero
        max_vals = (
            torch.max(torch.abs(environmental_rumble), dim=2, keepdim=True)[0] + 1e-8
        )
        environmental_rumble = environmental_rumble / max_vals

        # Reshape back to [C, T]
        return (environmental_rumble.view(num_channels, window_length) * amplitude).to(
            torch.float32
        )


def inject_snr_noise(clean_signal, target_snr_db):
    signal_power = torch.mean(clean_signal**2)
    if signal_power == 0:
        return clean_signal

    noise_power = signal_power / (10 ** (target_snr_db / 10))
    # randn_like automatically inherits the device of clean_signal
    noise = torch.randn_like(clean_signal) * torch.sqrt(noise_power)
    return (clean_signal + noise).to(torch.float32)


def augment_batch(batch_tensor, snr_range=(10, 30)):
    batch_size = batch_tensor.shape[0]
    # Use the batch's dynamic device so it scales whether on CPU or GPU
    random_snrs = torch.zeros(batch_size, device=batch_tensor.device).uniform_(
        snr_range[0], snr_range[1]
    )

    signal_powers = torch.mean(batch_tensor**2, dim=(1, 2), keepdim=True)
    noise_powers = signal_powers / (10 ** (random_snrs.view(-1, 1, 1) / 10))

    noise = torch.randn_like(batch_tensor) * torch.sqrt(noise_powers)
    return (batch_tensor + noise).to(torch.float32)
