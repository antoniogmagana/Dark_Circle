"""
Pytest configuration and shared fixtures for Dark Circle tests.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace


@pytest.fixture
def mock_config():
    """Create a mock configuration object for testing."""
    config = SimpleNamespace()
    
    # Hardware
    config.DEVICE = torch.device("cpu")
    
    # Training hyperparameters
    config.BATCH_SIZE = 32
    config.EPOCHS = 10
    config.NUM_WORKERS = 2
    config.LOG_INTERVAL = 10
    config.BLOCK_SIZE = 60
    config.USABLE_SIZE = 45
    config.LEARNING_RATE = 0.001
    
    # Database params (mock)
    config.DB_CONN_PARAMS = {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "host": "localhost",
        "port": 5432,
    }
    
    # Training mode
    config.TRAINING_MODE = "detection"
    config.INSTANCE_SEED = 0
    config.BEST_MODEL_METRIC = "val_f1"
    
    # Dataset configuration
    config.TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]
    config.TRAIN_SENSORS = ["audio", "seismic"]
    config.IN_CHANNELS = 2
    config.ACOUSTIC_SR = 16000
    config.REF_SAMPLE_RATE = 16000
    config.SAMPLE_SECONDS = 3.0
    
    # Native sample rates
    config.NATIVE_SR = {
        "iobt": {"audio": 16000, "seismic": 100, "accel": 100},
        "focal": {"audio": 16000, "seismic": 100, "accel": 100},
        "m3nvc": {"audio": 1600, "seismic": 200, "accel": 200},
    }
    
    # Class mapping
    config.CLASS_MAP = {0: "background", 1: "light", 2: "heavy"}
    config.INSTANCE_TO_CLASS = {}
    
    # Model architecture parameters (for both DetectionCNN and ClassificationCNN)
    config.CHANNELS = [32, 64, 128, 256]  # Used by both models
    config.KERNELS = [5, 3]  # DetectionCNN uses list of ints
    config.STRIDES = [2, 1]  # DetectionCNN uses list of ints
    config.PADS = [2, 1]  # DetectionCNN uses list of ints
    config.KERNEL = 3  # ClassificationCNN uses single int
    config.HIDDEN = 128
    config.DROPOUT = 0.5
    
    # Mel spectrogram parameters
    config.N_MELS = 64
    config.N_FFT = 512
    config.HOP_LENGTH = 256
    
    # Augmentation
    config.SYNTHESIZE_BACKGROUND = False
    config.SYNTHESIZE_PROBABILITY = 0.5
    
    # Directories
    config.CHECKPOINT_DIR = "./checkpoints"
    config.EVAL_RESULTS_DIR = "./eval_results"
    config.EVAL_STEPS = 200
    
    return config


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 48000)  # 2 channels, 3 seconds at 16kHz


@pytest.fixture
def sample_batch():
    """Create a batch of sample tensors."""
    return torch.randn(8, 2, 48000)  # Batch of 8


@pytest.fixture
def mock_db_connection(monkeypatch):
    """Mock database connection for testing."""
    class MockCursor:
        def __init__(self):
            self.query = None
            self.params = None
            
        def execute(self, query, params=None):
            self.query = query
            self.params = params
            
        def fetchone(self):
            return (0.0, 100.0)  # Mock time bounds
            
        def fetchall(self):
            return [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7)]
            
        def close(self):
            pass
    
    class MockConnection:
        def __init__(self):
            self.autocommit = True
            self.cursor_obj = MockCursor()
            
        def cursor(self):
            return self.cursor_obj
            
        def close(self):
            pass
    
    return MockConnection()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model testing."""
    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # 3 seconds at 16kHz
    duration = 3.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate a simple sine wave
    t = np.linspace(0, duration, num_samples)
    frequency = 440  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    return torch.tensor(audio, dtype=torch.float32)


@pytest.fixture
def sample_seismic_data():
    """Generate sample seismic data for testing."""
    # 3 seconds at 100Hz
    duration = 3.0
    sample_rate = 100
    num_samples = int(duration * sample_rate)
    
    # Generate low-frequency noise
    seismic = np.random.randn(num_samples) * 0.1
    
    return torch.tensor(seismic, dtype=torch.float32)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
