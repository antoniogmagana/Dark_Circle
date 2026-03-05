import numpy as np
from scipy import signal

# ==========================================
# 1. Negative Class Generation (No Vehicle)
# ==========================================

def generate_white_noise(window_length=16000, amplitude=0.01):
    """
    Generates a pure thermal/electronic noise baseline using a Gaussian distribution.
    Simulates the internal hiss of a microphone or geophone sitting in dead silence.
    """
    noise = np.random.randn(window_length) * amplitude
    return noise.astype(np.float32)

def generate_no_vehicle_sample(window_length=16000, noise_profile="environmental", amplitude=0.01):
    """
    Generates a realistic 'No Vehicle' (Class 0) 1-second window.
    
    noise_profile options:
    - 'sensor_hiss': Pure white noise (flat frequency response).
    - 'environmental': Low-pass filtered noise simulating wind or distant rumble.
    """
    # Generate the base thermal noise (standard normal distribution)
    base_noise = np.random.randn(window_length)
    
    if noise_profile == "sensor_hiss":
        return (base_noise * amplitude).astype(np.float32)
        
    elif noise_profile == "environmental":
        # Create a Low-Pass Butterworth filter to strip high-pitch hiss
        # and keep the low-frequency rumble (mimics wind or ground vibrations)
        b, a = signal.butter(N=2, Wn=0.05, btype='lowpass')
        
        # Apply the filter to the noise
        environmental_rumble = signal.filtfilt(b, a, base_noise)
        
        # Normalize and scale to the requested amplitude
        environmental_rumble = (environmental_rumble / np.max(np.abs(environmental_rumble)))
        return (environmental_rumble * amplitude).astype(np.float32)

# ==========================================
# 2. Data Augmentation & Utility
# ==========================================

def inject_snr_noise(clean_signal, target_snr_db):
    """
    Injects realistic Gaussian noise into a clean vehicle signal based on a strict 
    Signal-to-Noise Ratio (SNR) in Decibels. 
    """
    # Calculate the signal power (Mean Square)
    signal_power = np.mean(clean_signal ** 2)
    
    # Safety check for dead arrays (prevents divide-by-zero)
    if signal_power == 0:
        return clean_signal
        
    # Calculate target noise power based on the requested Decibel drop
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    
    # Generate the noise and add it to the clean signal
    noise = np.random.randn(len(clean_signal)) * np.sqrt(noise_power)
    noisy_signal = clean_signal + noise
    
    return noisy_signal.astype(np.float32)

def augment_batch(batch_matrix, snr_range=(10, 30)):
    """
    A wrapper function to apply random noise injection to an entire training batch.
    batch_matrix: Shape (batch_size, window_length)
    """
    augmented_batch = np.zeros_like(batch_matrix)
    
    for i in range(batch_matrix.shape[0]):
        # Pick a random SNR between the specified range (e.g., 10dB to 30dB)
        random_snr = np.random.uniform(snr_range[0], snr_range[1])
        augmented_batch[i] = inject_snr_noise(batch_matrix[i], random_snr)
        
    return augmented_batch

def upsample_seismic_fft(seismic_array, original_sr, target_sr):
    """
    Upsamples a low-frequency signal (e.g., 100 Hz seismic) to a higher frequency 
    (e.g., 16000 Hz) using the Fourier Transform method to preserve harmonics.
    """
    if original_sr >= target_sr:
        return seismic_array
        
    # Calculate exactly how many points the new array needs
    duration = len(seismic_array) / original_sr
    target_length = int(duration * target_sr)
    
    # scipy.signal.resample perfectly preserves the original wave curves via FFT
    upsampled_array = signal.resample(seismic_array, target_length)
    
    return upsampled_array.astype(np.float32)