import torch
import config


def generate_white_noise(window_length=config.ACOUSTIC_SR, amplitude=0.01):
    noise = torch.randn(window_length, device=config.DEVICE) * amplitude
    return noise.to(torch.float32)


def generate_no_vehicle_sample(
    window_length=config.ACOUSTIC_SR, noise_profile="environmental", amplitude=0.01
):
    base_noise = torch.randn(window_length, device=config.DEVICE)

    if noise_profile == "sensor_hiss":
        return (base_noise * amplitude).to(torch.float32)

    elif noise_profile == "environmental":
        kernel = (
            torch.ones((1, 1, config.NOISE_KERNEL_SIZE), device=config.DEVICE)
            / config.NOISE_KERNEL_SIZE
        )
        noise_reshaped = base_noise.view(1, 1, -1)
        environmental_rumble = torch.nn.functional.conv1d(
            noise_reshaped, kernel, padding=config.NOISE_KERNEL_SIZE // 2
        )
        environmental_rumble = environmental_rumble / torch.max(
            torch.abs(environmental_rumble)
        )
        return (environmental_rumble.squeeze() * amplitude).to(torch.float32)


def inject_snr_noise(clean_signal, target_snr_db):
    signal_power = torch.mean(clean_signal**2)
    if signal_power == 0:
        return clean_signal

    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = torch.randn_like(clean_signal) * torch.sqrt(noise_power)
    return (clean_signal + noise).to(torch.float32)


def augment_batch(batch_tensor, snr_range=(10, 30)):
    batch_size = batch_tensor.shape[0]
    random_snrs = torch.zeros(batch_size, device=config.DEVICE).uniform_(
        snr_range[0], snr_range[1]
    )

    signal_powers = torch.mean(batch_tensor**2, dim=(1, 2), keepdim=True)
    noise_powers = signal_powers / (10 ** (random_snrs.view(-1, 1, 1) / 10))

    noise = torch.randn_like(batch_tensor) * torch.sqrt(noise_powers)
    return (batch_tensor + noise).to(torch.float32)
