"""
Tests for model-train/config.py
Tests configuration loading, validation, and constants.
"""
import pytest
import sys
import os
from pathlib import Path

# Set DB password before importing config to avoid input prompt
os.environ['DB_PASSWORD'] = 'test_password'

# Add model-train to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-train"))

import config


class TestDeviceConfiguration:
    """Test device configuration logic."""
    
    def test_device_is_valid(self):
        """Test that DEVICE is a valid torch device."""
        assert hasattr(config, 'DEVICE')
        assert str(config.DEVICE) in ['cpu', 'cuda', 'mps']
    
    def test_device_available(self):
        """Test device corresponds to available hardware."""
        import torch
        device_str = str(config.DEVICE).split(':')[0]
        
        if device_str == 'cuda':
            assert torch.cuda.is_available()
        elif device_str == 'mps':
            assert torch.backends.mps.is_available()
        else:
            assert device_str == 'cpu'


class TestHyperparameters:
    """Test training hyperparameters."""
    
    def test_batch_size_positive(self):
        """Test batch size is positive."""
        assert config.BATCH_SIZE > 0
        assert isinstance(config.BATCH_SIZE, int)
    
    def test_epochs_positive(self):
        """Test epochs is positive."""
        assert config.EPOCHS > 0
        assert isinstance(config.EPOCHS, int)
    
    def test_num_workers_valid(self):
        """Test num_workers is non-negative."""
        assert config.NUM_WORKERS >= 0
        assert isinstance(config.NUM_WORKERS, int)
    
    def test_block_size_valid(self):
        """Test block size configuration."""
        assert config.BLOCK_SIZE > 0
        assert config.USABLE_SIZE > 0
        assert config.USABLE_SIZE <= config.BLOCK_SIZE
    
    def test_log_interval_positive(self):
        """Test log interval is positive."""
        assert config.LOG_INTERVAL > 0


class TestDatabaseConfiguration:
    """Test database configuration."""
    
    def test_db_params_structure(self):
        """Test DB_CONN_PARAMS has required fields."""
        assert hasattr(config, 'DB_CONN_PARAMS')
        assert isinstance(config.DB_CONN_PARAMS, dict)
        
        required_keys = ['dbname', 'user', 'password', 'host', 'port']
        for key in required_keys:
            assert key in config.DB_CONN_PARAMS
    
    def test_db_port_valid(self):
        """Test database port is valid."""
        port = config.DB_CONN_PARAMS['port']
        assert isinstance(port, int)
        assert 1 <= port <= 65535


class TestTrainingMode:
    """Test training mode configuration."""
    
    def test_training_mode_valid(self):
        """Test TRAINING_MODE is one of valid options."""
        assert hasattr(config, 'TRAINING_MODE')
        valid_modes = ['detection', 'category', 'instance']
        assert config.TRAINING_MODE in valid_modes
    
    def test_instance_seed_valid(self):
        """Test instance seed is non-negative."""
        assert hasattr(config, 'INSTANCE_SEED')
        assert config.INSTANCE_SEED >= 0


class TestModelSelection:
    """Test model selection criteria."""
    
    def test_best_model_metric_valid(self):
        """Test BEST_MODEL_METRIC is valid."""
        valid_metrics = [
            'val_acc', 'val_loss', 'val_f1', 
            'val_precision', 'val_recall'
        ]
        assert config.BEST_MODEL_METRIC in valid_metrics


class TestDatasetConfiguration:
    """Test dataset and sensor configuration."""
    
    def test_train_datasets_valid(self):
        """Test TRAIN_DATASETS contains valid datasets."""
        assert hasattr(config, 'TRAIN_DATASETS')
        assert isinstance(config.TRAIN_DATASETS, list)
        assert len(config.TRAIN_DATASETS) > 0
        
        valid_datasets = ['iobt', 'focal', 'm3nvc']
        for ds in config.TRAIN_DATASETS:
            assert ds in valid_datasets
    
    def test_train_sensors_valid(self):
        """Test TRAIN_SENSORS contains valid sensors."""
        assert hasattr(config, 'TRAIN_SENSORS')
        assert isinstance(config.TRAIN_SENSORS, list)
        assert len(config.TRAIN_SENSORS) > 0
        
        valid_sensors = ['audio', 'seismic', 'accel']
        for sensor in config.TRAIN_SENSORS:
            assert sensor in valid_sensors
    
    def test_in_channels_matches_sensors(self):
        """Test IN_CHANNELS matches sensor configuration."""
        expected_channels = len(config.TRAIN_SENSORS)
        if 'accel' in config.TRAIN_SENSORS:
            # Accel adds 2 extra channels (3 total for accel vs 1 for others)
            expected_channels += 2
        
        assert config.IN_CHANNELS == expected_channels
    
    def test_acoustic_sr_positive(self):
        """Test acoustic sample rate is positive."""
        assert config.ACOUSTIC_SR > 0
        assert isinstance(config.ACOUSTIC_SR, int)
    
    def test_native_sr_structure(self):
        """Test NATIVE_SR dictionary structure."""
        assert hasattr(config, 'NATIVE_SR')
        assert isinstance(config.NATIVE_SR, dict)
        
        # Check each dataset has sensor mappings
        for dataset in config.TRAIN_DATASETS:
            assert dataset in config.NATIVE_SR
            for sensor in ['audio', 'seismic', 'accel']:
                assert sensor in config.NATIVE_SR[dataset]
                assert config.NATIVE_SR[dataset][sensor] > 0
    
    def test_ref_sample_rate_is_maximum(self):
        """Test REF_SAMPLE_RATE is the maximum native sample rate."""
        max_native = max(
            config.NATIVE_SR[ds][s]
            for ds in config.TRAIN_DATASETS
            for s in config.TRAIN_SENSORS
        )
        assert config.REF_SAMPLE_RATE == max_native


class TestDirectoryConfiguration:
    """Test directory paths."""
    
    def test_checkpoint_dir_set(self):
        """Test checkpoint directory is configured."""
        assert hasattr(config, 'CHECKPOINT_DIR')
        assert isinstance(config.CHECKPOINT_DIR, str)
        assert len(config.CHECKPOINT_DIR) > 0
    
    def test_eval_results_dir_set(self):
        """Test evaluation results directory is configured."""
        assert hasattr(config, 'EVAL_RESULTS_DIR')
        assert isinstance(config.EVAL_RESULTS_DIR, str)
        assert len(config.EVAL_RESULTS_DIR) > 0
    
    def test_eval_steps_positive(self):
        """Test eval steps is positive."""
        assert config.EVAL_STEPS > 0


class TestConfigConsistency:
    """Test overall configuration consistency."""
    
    def test_no_missing_critical_attributes(self):
        """Test all critical configuration attributes exist."""
        critical_attrs = [
            'DEVICE', 'BATCH_SIZE', 'EPOCHS', 'NUM_WORKERS',
            'TRAIN_DATASETS', 'TRAIN_SENSORS', 'IN_CHANNELS',
            'TRAINING_MODE', 'DB_CONN_PARAMS', 'NATIVE_SR',
            'REF_SAMPLE_RATE', 'ACOUSTIC_SR'
        ]
        
        for attr in critical_attrs:
            assert hasattr(config, attr), f"Missing critical attribute: {attr}"
    
    def test_config_immutable_types(self):
        """Test configuration uses appropriate immutable types where needed."""
        # Lists should be lists (will be validated, not modified)
        assert isinstance(config.TRAIN_DATASETS, list)
        assert isinstance(config.TRAIN_SENSORS, list)
        
        # Dicts should be dicts
        assert isinstance(config.DB_CONN_PARAMS, dict)
        assert isinstance(config.NATIVE_SR, dict)
