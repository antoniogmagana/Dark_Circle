"""
Tests for model-train/data_generator.py
Tests data generation, noise injection, and augmentation functions.
"""
import pytest
import torch
import sys
import os
from pathlib import Path

# Set DB password to avoid input prompt
os.environ['DB_PASSWORD'] = 'test_password'

# Add model-train to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-train"))

from data_generator import (
    generate_white_noise,
    generate_no_vehicle_sample,
    inject_snr_noise,
    augment_batch
)


class TestGenerateWhiteNoise:
    """Test white noise generation."""
    
    def test_basic_generation(self, mock_config):
        """Test basic white noise generation."""
        noise = generate_white_noise(mock_config)
        
        assert isinstance(noise, torch.Tensor)
        assert noise.dtype == torch.float32
        assert noise.ndim == 2  # [channels, time]
    
    def test_custom_window_length(self, mock_config):
        """Test with custom window length."""
        window_length = 1000
        noise = generate_white_noise(mock_config, window_length=window_length)
        
        assert noise.shape[1] == window_length
    
    def test_custom_channels(self, mock_config):
        """Test with custom number of channels."""
        num_channels = 3
        noise = generate_white_noise(mock_config, num_channels=num_channels)
        
        assert noise.shape[0] == num_channels
    
    def test_amplitude_control(self, mock_config):
        """Test amplitude parameter controls noise level."""
        amplitude = 0.05
        noise = generate_white_noise(mock_config, amplitude=amplitude)
        
        # Noise should be roughly within expected range
        std = noise.std().item()
        assert 0.9 * amplitude < std < 1.1 * amplitude
    
    def test_output_on_cpu(self, mock_config):
        """Test output is on CPU as specified in comments."""
        noise = generate_white_noise(mock_config)
        assert noise.device.type == 'cpu'
    
    def test_shape_with_config_defaults(self, mock_config):
        """Test shape matches config defaults."""
        expected_length = int(mock_config.REF_SAMPLE_RATE * mock_config.SAMPLE_SECONDS)
        noise = generate_white_noise(mock_config)
        
        assert noise.shape[0] == mock_config.IN_CHANNELS
        assert noise.shape[1] == expected_length


class TestGenerateNoVehicleSample:
    """Test background/no-vehicle sample generation."""
    
    def test_sensor_hiss_profile(self, mock_config):
        """Test sensor_hiss noise profile."""
        sample = generate_no_vehicle_sample(
            mock_config,
            noise_profile="sensor_hiss"
        )
        
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.ndim == 2
    
    def test_environmental_profile(self, mock_config):
        """Test environmental noise profile."""
        sample = generate_no_vehicle_sample(
            mock_config,
            noise_profile="environmental"
        )
        
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.ndim == 2
    
    def test_custom_amplitude_scalar(self, mock_config):
        """Test with scalar amplitude."""
        amplitude = 0.02
        sample = generate_no_vehicle_sample(
            mock_config,
            amplitude=amplitude,
            noise_profile="sensor_hiss"
        )
        
        # Check amplitude is in reasonable range
        assert sample.abs().mean() > 0
        assert sample.abs().max() < amplitude * 10  # Sanity check
    
    def test_custom_amplitude_tensor(self, mock_config):
        """Test with tensor amplitude (per-channel)."""
        num_channels = mock_config.IN_CHANNELS
        amplitude = torch.tensor([0.01, 0.02])[:num_channels]
        
        sample = generate_no_vehicle_sample(
            mock_config,
            amplitude=amplitude,
            noise_profile="sensor_hiss"
        )
        
        assert sample.shape[0] == num_channels
    
    def test_environmental_smoother_than_hiss(self, mock_config):
        """Test environmental noise is smoother than sensor hiss."""
        hiss = generate_no_vehicle_sample(
            mock_config,
            noise_profile="sensor_hiss",
            amplitude=0.01
        )
        
        env = generate_no_vehicle_sample(
            mock_config,
            noise_profile="environmental",
            amplitude=0.01
        )
        
        # Environmental should have lower high-frequency content
        # Calculate approximate frequency content via diff
        hiss_diff = torch.diff(hiss, dim=1).abs().mean()
        env_diff = torch.diff(env, dim=1).abs().mean()
        
        assert env_diff < hiss_diff
    
    def test_output_shape(self, mock_config):
        """Test output shape matches configuration."""
        expected_length = int(mock_config.REF_SAMPLE_RATE * mock_config.SAMPLE_SECONDS)
        sample = generate_no_vehicle_sample(mock_config)
        
        assert sample.shape == (mock_config.IN_CHANNELS, expected_length)
    
    def test_output_on_cpu(self, mock_config):
        """Test output is on CPU."""
        sample = generate_no_vehicle_sample(mock_config)
        assert sample.device.type == 'cpu'


class TestInjectSNRNoise:
    """Test SNR-based noise injection."""
    
    def test_basic_injection(self):
        """Test basic SNR noise injection."""
        clean_signal = torch.randn(2, 1000)
        target_snr = 20
        
        noisy_signal = inject_snr_noise(clean_signal, target_snr)
        
        assert isinstance(noisy_signal, torch.Tensor)
        assert noisy_signal.shape == clean_signal.shape
        assert noisy_signal.dtype == torch.float32
    
    def test_higher_snr_less_noise(self):
        """Test that higher SNR results in less noise."""
        clean_signal = torch.ones(2, 1000)  # Constant signal
        
        noisy_low_snr = inject_snr_noise(clean_signal, target_snr_db=10)
        noisy_high_snr = inject_snr_noise(clean_signal, target_snr_db=30)
        
        # Higher SNR should be closer to original
        error_low = (noisy_low_snr - clean_signal).abs().mean()
        error_high = (noisy_high_snr - clean_signal).abs().mean()
        
        assert error_high < error_low
    
    def test_zero_signal_handling(self):
        """Test handling of zero signal."""
        zero_signal = torch.zeros(2, 1000)
        
        result = inject_snr_noise(zero_signal, target_snr_db=20)
        
        # Should return the original signal unchanged
        assert torch.allclose(result, zero_signal)
    
    def test_preserves_dc_offset(self):
        """Test that DC offset is preserved."""
        # Signal with DC offset
        signal = torch.ones(2, 1000) * 5.0
        
        noisy = inject_snr_noise(signal, target_snr_db=20)
        
        # Mean should be close to original (DC preserved)
        assert abs(noisy.mean().item() - 5.0) < 0.5
    
    def test_device_inheritance(self):
        """Test that output inherits device from input."""
        signal_cpu = torch.randn(2, 1000)
        result = inject_snr_noise(signal_cpu, target_snr_db=20)
        
        assert result.device == signal_cpu.device
    
    def test_approximate_snr(self):
        """Test that actual SNR is approximately correct."""
        # Create a clean signal with known power
        signal = torch.randn(2, 10000)  # Large sample for better statistics
        target_snr_db = 20
        
        noisy = inject_snr_noise(signal, target_snr_db)
        
        # Calculate actual SNR
        ac_signal = signal - signal.mean()
        signal_power = (ac_signal ** 2).mean()
        
        noise = noisy - signal
        noise_power = (noise ** 2).mean()
        
        actual_snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        
        # Should be within 2dB of target
        assert abs(actual_snr_db.item() - target_snr_db) < 2.0


class TestAugmentBatch:
    """Test batch augmentation with random SNR."""
    
    def test_basic_augmentation(self):
        """Test basic batch augmentation."""
        batch = torch.randn(8, 2, 1000)
        
        augmented = augment_batch(batch, snr_range=(10, 30))
        
        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == batch.shape
        assert augmented.dtype == torch.float32
    
    def test_different_snr_per_sample(self):
        """Test that different samples get different SNR."""
        # Use constant signal to see noise differences
        batch = torch.ones(8, 2, 1000)
        
        augmented = augment_batch(batch, snr_range=(10, 30))
        
        # Different samples should have different noise levels
        sample_vars = augmented.var(dim=(1, 2))
        
        # At least some variance in the variances
        assert sample_vars.std() > 0
    
    def test_snr_range_effect(self):
        """Test that SNR range affects output."""
        batch = torch.ones(16, 2, 1000)
        
        # Low SNR range (more noise)
        aug_low = augment_batch(batch, snr_range=(5, 10))
        
        # High SNR range (less noise)
        aug_high = augment_batch(batch, snr_range=(25, 30))
        
        # Low SNR should deviate more from original
        error_low = (aug_low - batch).abs().mean()
        error_high = (aug_high - batch).abs().mean()
        
        assert error_low > error_high
    
    def test_preserves_batch_dc_offsets(self):
        """Test that DC offsets are preserved per sample."""
        # Create batch with different DC offsets
        batch = torch.randn(8, 2, 1000)
        batch_means = batch.mean(dim=(1, 2), keepdim=True)
        
        augmented = augment_batch(batch, snr_range=(15, 25))
        augmented_means = augmented.mean(dim=(1, 2), keepdim=True)
        
        # Means should be close
        mean_diff = (batch_means - augmented_means).abs().mean()
        assert mean_diff < 0.5
    
    def test_device_preservation(self):
        """Test that output stays on same device as input."""
        batch_cpu = torch.randn(4, 2, 1000)
        result = augment_batch(batch_cpu, snr_range=(10, 30))
        
        assert result.device == batch_cpu.device
    
    def test_single_sample_batch(self):
        """Test with single sample batch."""
        batch = torch.randn(1, 2, 1000)
        
        augmented = augment_batch(batch, snr_range=(15, 25))
        
        assert augmented.shape == batch.shape
    
    def test_large_batch(self):
        """Test with large batch."""
        batch = torch.randn(128, 2, 1000)
        
        augmented = augment_batch(batch, snr_range=(10, 30))
        
        assert augmented.shape == batch.shape


class TestDataGeneratorIntegration:
    """Integration tests for data generator functions."""
    
    def test_noise_and_augmentation_pipeline(self, mock_config):
        """Test combining noise generation and augmentation."""
        # Generate background samples
        samples = [
            generate_no_vehicle_sample(mock_config, noise_profile="environmental")
            for _ in range(8)
        ]
        
        # Stack into batch
        batch = torch.stack(samples)
        
        # Augment
        augmented = augment_batch(batch, snr_range=(15, 25))
        
        assert augmented.shape == batch.shape
    
    def test_consistent_output_shapes(self, mock_config):
        """Test all generators produce consistent shapes."""
        white_noise = generate_white_noise(mock_config)
        no_vehicle = generate_no_vehicle_sample(mock_config)
        
        assert white_noise.shape == no_vehicle.shape
    
    def test_multiple_noise_profiles_compatible(self, mock_config):
        """Test different noise profiles are compatible."""
        hiss = generate_no_vehicle_sample(
            mock_config, 
            noise_profile="sensor_hiss"
        )
        env = generate_no_vehicle_sample(
            mock_config,
            noise_profile="environmental"
        )
        
        assert hiss.shape == env.shape
        assert hiss.dtype == env.dtype
