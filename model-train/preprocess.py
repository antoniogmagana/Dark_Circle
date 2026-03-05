import numpy as np
import librosa
import torch
import torchaudio.transforms as T

def zero_center_window(data_matrix):
    """
    Subtracts the local mean from each window to remove DC offset and sensor drift.
    Input shape: (num_samples, window_length)
    """
    local_means = np.mean(data_matrix, axis=-1, keepdims=True)
    return data_matrix - local_means

def standardize_global(centered_data, global_std):
    """
    Scales data by the global standard deviation of the training dataset.
    Input shape: (num_samples, window_length)
    """
    safe_std = np.where(global_std == 0, 1e-6, global_std)
    return centered_data / safe_std

def calculate_smv(axis_x, axis_y, axis_z):
    """
    Calculates the Signal Magnitude Vector (SMV) for any 3-axis sensor data 
    (accelerometer or 3-axis geophone). Returns a rotation-invariant 1D array.
    """
    smv = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
    return zero_center_window(smv)

def extract_mel_spectrogram_batch_cpu(signal_batch, sample_rate=16000, n_mels=64, hop_length=512):
    """
    CPU-bound Mel-Spectrogram generation using librosa (NumPy arrays).
    Best for pre-generating static dataset files before training begins.
    """
    num_samples = signal_batch.shape[0]
    
    # Process the first sample to dynamically determine the time-step dimension
    sample_mel = librosa.feature.melspectrogram(
        y=signal_batch[0], sr=sample_rate, n_mels=n_mels, hop_length=hop_length
    )
    mel_shape = sample_mel.shape
    
    # Pre-allocate the 3D tensor to save RAM
    mel_tensor = np.zeros((num_samples, mel_shape[0], mel_shape[1]), dtype=np.float32)
    
    for i in range(num_samples):
        S = librosa.feature.melspectrogram(
            y=signal_batch[i], sr=sample_rate, n_mels=n_mels, hop_length=hop_length
        )
        mel_tensor[i] = librosa.power_to_db(S, ref=np.max)
        
    return mel_tensor

def extract_mel_spectrogram_batch_gpu(signal_tensor, target_device, sample_rate=16000, n_mels=64, hop_length=512):
    """
    Hardware-agnostic Mel-Spectrogram generation using PyTorch (Tensors).
    Automatically routes to Mac M2 (MPS) or Ubuntu Server (CUDA) via target_device.
    """
    # Initialize transforms and send them to the target hardware
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length
    ).to(target_device)
    
    amplitude_to_db = T.AmplitudeToDB(top_db=80).to(target_device)
    
    # Process the entire batch instantly without a loop
    mel_power = mel_spectrogram_transform(signal_tensor)
    mel_db = amplitude_to_db(mel_power)
    
    return mel_db