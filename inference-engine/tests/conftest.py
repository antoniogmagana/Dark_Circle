"""
Shared pytest fixtures for inference-engine tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Buffer-level test fixtures assume per-channel native rates (seismic=100Hz,
# accel=100Hz). Force NATIVE_RATES=1 before any ingestor module is imported
# so SensorBuffer doesn't upsample everything to TARGET_RATE.
os.environ.setdefault("NATIVE_RATES", "1")

# Mirror the container layout: each node ships flat into /app, so its
# siblings are imported via top-level names (``from whitelist import ...``,
# ``from dispatch import ...``). Put each per-node directory on sys.path
# so tests resolve those names the same way the deployed images do.
INFERENCE_ENGINE_ROOT = Path(__file__).parent.parent
SRC_PATH = INFERENCE_ENGINE_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
for node_dir in SRC_PATH.iterdir():
    if node_dir.is_dir():
        sys.path.insert(0, str(node_dir))

# Import protobuf definitions
try:
    from google.protobuf.timestamp_pb2 import Timestamp as ProtoTimestamp
    from inference_protos import inference_pb2

    PROTOS_AVAILABLE = True
except ImportError:
    PROTOS_AVAILABLE = False
    inference_pb2 = None
    ProtoTimestamp = None


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
    node.create_subscription = MagicMock(return_value=MagicMock())
    node.create_publisher = MagicMock(return_value=MagicMock())
    node.create_timer = MagicMock(return_value=MagicMock())

    # Mock get_topic_names_and_types for Discovery node testing
    node.get_topic_names_and_types.return_value = [
        ("/sensor_array_01/acoustic", ["ros2_interfaces/msg/RawSensorReading"]),
        ("/sensor_array_01/seismic", ["ros2_interfaces/msg/RawSensorReading"]),
    ]

    return node


@pytest.fixture
def mock_nats_client():
    """Mock async NATS client (with JetStream stub) for testing."""
    client = AsyncMock()

    # Mock async methods
    client.connect = AsyncMock()
    client.subscribe = AsyncMock(return_value=MagicMock())
    client.publish = AsyncMock()
    client.close = AsyncMock()

    # Mock connection state
    client.is_connected = True

    # Store published messages for verification in tests
    client._published_messages = []

    async def publish_spy(subject, data):
        client._published_messages.append({"subject": subject, "data": data})

    client.publish.side_effect = publish_spy

    # JetStream context: same shape (subscribe / publish) as the core API
    # so tests reach in through ``nc.jetstream()`` if they want.
    js_ctx = AsyncMock()
    js_ctx.subscribe = AsyncMock(return_value=MagicMock())
    js_ctx._published_messages = client._published_messages

    async def js_publish_spy(subject, data, **_kwargs):
        client._published_messages.append({"subject": subject, "data": data})

    js_ctx.publish = AsyncMock(side_effect=js_publish_spy)
    client.jetstream = MagicMock(return_value=js_ctx)

    return client


@pytest.fixture
def mock_k8s_client():
    """Mock Kubernetes API client for testing."""
    client = MagicMock()

    # Mock deployment operations
    client.create_namespaced_deployment = MagicMock()
    client.delete_namespaced_deployment = MagicMock()

    # Mock list_namespaced_deployment with realistic response
    mock_deployment = MagicMock()
    mock_deployment.metadata.labels = {
        "sensor-array": "sensor_array_01",
        "app": "ingestor",
    }

    mock_list_response = MagicMock()
    mock_list_response.items = [mock_deployment]
    client.list_namespaced_deployment.return_value = mock_list_response

    return client


@pytest.fixture
def mock_ros2_message():
    """Mock ROS2 RawSensorReading message."""
    msg = MagicMock()
    msg.sensor_id = "sensor_array_01.aud"
    msg.start_time = 1700000000.0
    msg.amplitude_readings = np.random.randint(-32768, 32767, size=100, dtype=np.int16).tolist()
    return msg


@pytest.fixture
def mock_ros2_inference_result():
    """Mock ROS2 InferenceResult message class."""
    InferenceResult = MagicMock()
    InferenceResult.return_value = MagicMock()
    return InferenceResult


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

    ts = ProtoTimestamp()
    ts.seconds = int(base_timestamp)
    ts.nanos = int((base_timestamp - int(base_timestamp)) * 1e9)

    payload = inference_pb2.SensorData(sensor_id=sensor_id, time_stamp=ts)

    # Add sample acoustic data
    acoustic_data = np.random.randn(16000).astype(np.float32)
    payload.channels.append("acoustic")
    payload.acoustic_data.CopyFrom(inference_pb2.Tensor(shape=[16000], data=acoustic_data.tolist()))

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
        ts = ProtoTimestamp()
        ts.seconds = int(unix_time)
        ts.nanos = int((unix_time - int(unix_time)) * 1e9)
        return ts

    return _create


# ============================================================================
# Parametrize Helpers
# ============================================================================


@pytest.fixture(params=["acoustic", "seismic", "accel_x", "accel_y", "accel_z"])
def channel_name(request):
    """Parametrize over all channel names."""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 0.999])
def timestamp_offset(request):
    """Parametrize over timestamps within a window."""
    return request.param
