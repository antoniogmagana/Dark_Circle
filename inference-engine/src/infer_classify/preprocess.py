import torch
import torchaudio

# NOTICE: global 'config' is no longer imported


# ============================================================
# 1. Vectorized Mel Spectrogram Extraction
# ============================================================

# Cache transforms to prevent re-initializing them every batch.
# Keyed by device and hyperparameters to support dynamic configs
# across multiple batch evaluation runs.
_mel_transforms = {}

def get_mel_transform(device, config):
    global _mel_transforms
    
    # Create a unique key based on the parameters that define the transform
    key = (
        device, 
        config.REF_SAMPLE_RATE, 
        config.N_FFT, 
        config.HOP_LENGTH, 
        config.MEL_BINS
    )
    
    if key not in _mel_transforms:
        _mel_transforms[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.REF_SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.MEL_BINS,
        ).to(device)
        
    return _mel_transforms[key]


def extract_mel_spectrogram(batch_tensor, config):
    """
    Converts 1D waveforms to 2D Mel Spectrograms.
    Optimized to process all channels and batches simultaneously without for-loops.

    Input: [B, C, T] -> Output: [B, C, MEL_BINS, FRAMES]
    """
    device = batch_tensor.device
    mel_transform = get_mel_transform(device, config)

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

def preprocess_for_training(batch_tensor, config):
    """
    Preprocessing pipeline for training and inference.
    Normalizes raw ADC counts to a physical amplitude range, removes DC drift,
    and limits transient spikes.
    """
    # 1. ADC SCALE NORMALIZATION (maps raw counts to [-1, 1] based on bit depth)
    adc_scales = torch.tensor(
        config.CHANNEL_ADC_SCALES, dtype=batch_tensor.dtype
    ).view(1, -1, 1).to(batch_tensor.device)
    batch_tensor = batch_tensor / adc_scales

    # 2. PER-WINDOW MEAN SUBTRACTION (DC drift correction)
    window_mean = batch_tensor.mean(dim=-1, keepdim=True)
    batch_tensor = batch_tensor - window_mean

    # 3. TRANSIENT SPIKE CLIPPING
    batch_tensor = torch.clamp(batch_tensor, min=-10.0, max=10.0)

    # 4. Convert to frequency domain
    if getattr(config, "USE_MEL", True):
        batch_tensor = extract_mel_spectrogram(batch_tensor, config)

    return batch_tensor