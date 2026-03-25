"""
Shared pytest fixtures for inference-engine tests.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

# Add src to Python path for imports
INFERENCE_ENGINE_ROOT = Path(__file__).parent.parent
SRC_PATH = INFERENCE_ENGINE_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import protobuf definitions
try:
    from inference_protos import inference_pb2
    PROTOS_AVAILABLE = True
except ImportError:
    PROTOS_AVAILABLE = False
    inference_pb2 = None


# ============================================================================
# Basic Fixtures
# ============================================================================

@pytest.fixture
def sensor_id():
    """Default sensor ID for testing."""
    return "sensor_array_01"


@pytest.fixture
def base_timestamp():
    """Base timestamp for testing (Unix epoch)."""
    return 0.0  # Use 0.0 to avoid floating point precision issues with large numbers


@pytest.fixture
def window_duration():
    """Default window duration in seconds."""
    return 1.0


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def acoustic_sample_data():
    """Generate sample acoustic data (16kHz, 100 samples = 6.25ms)."""
    return np.random.randint(-32768, 32767, size=100, dtype=np.int16)


@pytest.fixture
def seismic_sample_data():
    """Generate sample seismic data (100Hz, 10 samples = 100ms)."""
    return np.random.randint(-8388608, 8388607, size=10, dtype=np.int32)


@pytest.fixture
def accel_sample_data():
    """Generate sample accelerometer data (100Hz, 10 samples = 100ms)."""
    return np.random.randint(-8388608, 8388607, size=10, dtype=np.int32)


@pytest.fixture
def full_acoustic_window():
    """Full 1-second window of acoustic data (16000 samples @ 16kHz)."""
    return np.random.randint(-32768, 32767, size=16000, dtype=np.int16)


@pytest.fixture
def full_seismic_window():
    """Full 1-second window of seismic data (100 samples @ 100Hz)."""
    return np.random.randint(-8388608, 8388607, size=100, dtype=np.int32)


@pytest.fixture
def full_accel_window():
    """Full 1-second window of accelerometer data (100 samples @ 100Hz)."""
    return np.random.randint(-8388608, 8388607, size=100, dtype=np.int32)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_ros2_node():
    """Mock ROS2 node for testing."""
    node = MagicMock()
    node.get_logger.return_value = MagicMock()
    node.create_subscription = MagicMock()
    node.create_publisher = MagicMock()
    node.create_timer = MagicMock()
    return node


@pytest.fixture
def mock_nats_client():
    """Mock NATS client for testing."""
    client = MagicMock()
    client.connect = MagicMock()
    client.subscribe = MagicMock()
    client.publish = MagicMock()
    client.close = MagicMock()
    return client


@pytest.fixture
def mock_k8s_client():
    """Mock Kubernetes API client for testing."""
    client = MagicMock()
    client.create_namespaced_deployment = MagicMock()
    client.delete_namespaced_deployment = MagicMock()
    client.list_namespaced_deployment = MagicMock()
    return client


# ============================================================================
# Protobuf Fixtures
# ============================================================================

@pytest.fixture
def skip_if_no_protos():
    """Skip test if protobuf definitions are not available."""
    if not PROTOS_AVAILABLE:
        pytest.skip("inference_protos not available")


@pytest.fixture
def sample_sensor_data_proto(sensor_id, base_timestamp):
    """Create a sample SensorData protobuf message."""
    if not PROTOS_AVAILABLE:
        pytest.skip("inference_protos not available")
    
    ts = inference_pb2.google_dot_protobuf_dot_timestamp__pb2.Timestamp()
    ts.seconds = int(base_timestamp)
    ts.nanos = int((base_timestamp - int(base_timestamp)) * 1e9)
    
    payload = inference_pb2.SensorData(
        sensor_id=sensor_id,
        time_stamp=ts
    )
    
    # Add sample acoustic data
    acoustic_data = np.random.randn(16000).astype(np.float32)
    payload.channels.append('acoustic')
    payload.acoustic_data.CopyFrom(
        inference_pb2.Tensor(
            shape=[16000],
            data=acoustic_data.tolist()
        )
    )
    
    return payload


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_arrays_equal():
    """Helper to assert numpy arrays are equal within tolerance."""
    def _assert_equal(arr1, arr2, rtol=1e-5, atol=1e-8):
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
    return _assert_equal


@pytest.fixture
def create_timestamp_proto():
    """Factory to create protobuf Timestamp messages."""
    if not PROTOS_AVAILABLE:
        pytest.skip("inference_protos not available")
    
    def _create(unix_time: float):
        ts = inference_pb2.google_dot_protobuf_dot_timestamp__pb2.Timestamp()
        ts.seconds = int(unix_time)
        ts.nanos = int((unix_time - int(unix_time)) * 1e9)
        return ts
    
    return _create


# ============================================================================
# Parametrize Helpers
# ============================================================================

@pytest.fixture(params=['acoustic', 'seismic', 'accel_x', 'accel_y', 'accel_z'])
def channel_name(request):
    """Parametrize over all channel names."""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 0.999])
def timestamp_offset(request):
    """Parametrize over timestamps within a window."""
    return request.param
