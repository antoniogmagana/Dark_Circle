import db_utils
import variables
import psycopg2
import numpy as np
import io
import os
import smote
import torch
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate
    )


def generate_noise(num_samples=5400, sample_rate=16000, window_seconds=1.0, encoding=0, alpha_mu=1.0):
    """
    Generates a matrix of synthetic pink/red noise samples using vectorized RFFT.
    Returns: (num_samples, N + 2) array [ID, Encoding, Data...]
    """
    N = int(sample_rate * window_seconds)
    # Metadata + Data columns
    aud_blank = np.zeros((num_samples, N + 2), dtype=np.float32)
    aud_blank[:, 0] = np.arange(num_samples)
    aud_blank[:, 1] = encoding
    
    # Frequency mapping
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    safe_freqs = np.where(freqs == 0, 1e-6, freqs).reshape(1, -1)
    
    # Unique alpha per sample for texture randomization
    alphas = np.random.normal(alpha_mu, 0.1, size=(num_samples, 1))
    
    # Generate White Noise and transform
    white_noise = np.random.randn(num_samples, N).astype(np.float32)
    f_space = np.fft.rfft(white_noise, axis=-1)
    
    # Apply 1/f^(alpha/2) filter
    filter_matrix = 1.0 / (safe_freqs ** (alphas / 2.0))
    filter_matrix[:, 0] = 0  # Remove DC offset
    
    # Inverse transform
    audio_data = np.fft.irfft(f_space * filter_matrix, n=N, axis=-1)
    aud_blank[:, 2:] = audio_data.astype(np.float32)
    
    return aud_blank


def upsample_seismic_fft(data_matrix, target_rate=16000, original_rate=100):
    """
    Upsamples 100Hz data to 16kHz using Spectral Interpolation (Zero-Padding).
    Input: (num_samples, 100) matrix (raw data only, no metadata columns)
    Output: (num_samples, 16000) matrix
    """
    num_samples, N_old = data_matrix.shape
    N_new = int(N_old * (target_rate / original_rate))
    
    # 1. Transform 100Hz data to Frequency Domain
    f_space = np.fft.rfft(data_matrix, axis=-1)
    
    # 2. Calculate padding to reach 16kHz (8001 bins)
    # Original bins: (100/2)+1 = 51. Target bins: (16000/2)+1 = 8001.
    target_bins = (target_rate // 2) + 1
    current_bins = f_space.shape[1]
    pad_width = target_bins - current_bins
    
    # Pad the end of the frequency array with zeros (High Frequencies)
    f_padded = np.pad(f_space, ((0, 0), (0, pad_width)), mode='constant')
    
    # 3. Inverse RFFT and scale amplitude by the upsampling ratio
    upsampled = np.fft.irfft(f_padded, n=N_new, axis=-1) * (N_new / N_old)
    
    return upsampled.astype(np.float32)




# ### Phase 1: Data Generation & Alignment (Completed Concepts)

# * [ ] **Generate Acoustic Noise:** Create 1-second windows of 16 kHz Pink Noise ($1/f$) using the vectorized generator.
# * [ ] **Generate Seismic Noise:** Create 1-second windows of 100 Hz Brownian/Red Noise ($1/f^2$) using the same generator.
# * [ ] **Spectral Upsampling:** Run both your *synthetic* 100 Hz noise and your *real* 100 Hz vehicle recordings through the `upsample_seismic_fft` function to sync everything to 16 kHz.
# * [ ] **SNR Injection:** Scale your synthetic noise amplitude to be a randomized fraction (e.g., 5% to 25%) of the standard deviation of your real vehicle signals to simulate realistic background environments.

# ### Phase 2: Signal Normalization (Completed Concepts)

# * [ ] **Zero-Centering (Window Level):** Subtract the local mean of each individual 1-second window to eliminate sensor drift and DC offset.
# * [ ] **Standardization (Global Level):** Divide by the global standard deviation for each specific sensor channel so the microphone, geophone, and accelerometer are on a mathematically level playing field.

# ### Phase 3: Feature Engineering (Next Steps)

# * [ ] **Acoustic Features:** Convert the normalized 1D acoustic arrays into 2D Mel-Spectrograms.
# * [ ] **Accelerometer Features:** Calculate the Signal Magnitude Vector (SMV) for the X, Y, and Z axes to make the reading rotation-invariant.
# * [ ] **Formatting:** Shape the inputs for the specific architectures (2D for CNN, sequential 1D for LSTM, multi-channel 1D for miniROCKET).

# ### Phase 4: Model Training & Evaluation

# * [ ] **Train the 3x3 Matrix:** Train the CNN, LSTM, and miniROCKET independently on each sensor type.
# * [ ] **Evaluate:** Score every model combination strictly using **Precision** (to avoid false alarms on your synthetic noise) and **Recall** (to ensure you don't miss heavy vehicles).

# ### Phase 5: The Polling Engine (Late Fusion)

# * [ ] **Detection Gate:** Average the probabilities from the models to make a binary decision (Vehicle Present vs. Background Noise).
# * [ ] **Classification Vote:** If a vehicle is detected, use a weighted vote from your top-performing models to determine the final weight class.


