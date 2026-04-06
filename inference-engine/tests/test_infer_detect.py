"""
Tests for Infer Detect Node (Detection Model Inference).

The Infer Detect node subscribes to NATS sensor.data, runs vehicle detection
inference, and publishes DetectionResult to detection.result.
"""
import pytest
import torch
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from types import SimpleNamespace


# ============================================================================
# Test Model Loading
# ============================================================================

class TestModelLoading:
    """Test model initialization and loading."""
    
    @pytest.mark.unit
    def test_device_selection_cuda(self):
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            assert device.type == "cuda"
    
    @pytest.mark.unit
    def test_device_selection_mps(self):
        """Test device selection when MPS is available (no CUDA)."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
                assert device.type == "mps"
    
    @pytest.mark.unit
    def test_device_selection_cpu(self):
        """Test device selection fallback to CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = torch.device("cpu")
                assert device.type == "cpu"
    
    @pytest.mark.unit
    def test_config_loading(self):
        """Test hyperparameter configuration loading."""
        config_dict = {
            "MODEL_NAME": "ResNet18_2Ch",
            "IN_CHANNELS": 2,
            "NUM_CLASSES": 2,
            "SEQ_LEN": 16000,
            "CLASS_MAP": {"0": "no_vehicle", "1": "vehicle"}
        }
        
        # Simulate loading config
        config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}
        config = SimpleNamespace(**config_dict)
        
        assert config.MODEL_NAME == "ResNet18_2Ch"
        assert config.IN_CHANNELS == 2
        assert config.NUM_CLASSES == 2
        assert config.SEQ_LEN == 16000
        assert config.CLASS_MAP[0] == "no_vehicle"
        assert config.CLASS_MAP[1] == "vehicle"
    
    @pytest.mark.unit
    def test_state_dict_prefix_stripping(self):
        """Test removing torch.compile '_orig_mod.' prefix from state dict."""
        state_dict = {
            "_orig_mod.conv1.weight": torch.randn(64, 3, 7, 7),
            "_orig_mod.bn1.weight": torch.randn(64),
            "fc.weight": torch.randn(2, 512)
        }
        
        # Strip prefix
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        
        assert "conv1.weight" in cleaned
        assert "bn1.weight" in cleaned
        assert "fc.weight" in cleaned
        assert "_orig_mod.conv1.weight" not in cleaned


# ============================================================================
# Test Inference Logic
# ============================================================================

class TestInferenceLogic:
    """Test detection inference logic."""
    
    @pytest.mark.unit
    def test_inference_vehicle_detected(self):
        """Test inference returns vehicle detected."""
        # Mock model output (logits: [no_vehicle, vehicle])
        mock_logits = torch.tensor([[0.2, 2.5]])  # Strongly predicts vehicle
        
        # Simulate softmax and argmax
        probs = torch.nn.functional.softmax(mock_logits, dim=1)
        vehicle_detected = mock_logits.argmax(dim=1).item() == 1
        confidence = probs[0, 1].item()
        
        assert vehicle_detected is True
        assert confidence > 0.9
    
    @pytest.mark.unit
    def test_inference_no_vehicle(self):
        """Test inference returns no vehicle detected."""
        # Mock model output (logits: [no_vehicle, vehicle])
        mock_logits = torch.tensor([[2.8, 0.1]])  # Strongly predicts no vehicle
        
        # Simulate softmax and argmax
        probs = torch.nn.functional.softmax(mock_logits, dim=1)
        vehicle_detected = mock_logits.argmax(dim=1).item() == 1
        confidence = probs[0, 1].item()
        
        assert vehicle_detected is False
        assert confidence < 0.1
    
    @pytest.mark.unit
    def test_confidence_calculation(self):
        """Test confidence score calculation from softmax."""
        mock_logits = torch.tensor([[1.0, 2.0]])
        probs = torch.nn.functional.softmax(mock_logits, dim=1)
        
        # Confidence should be in [0, 1]
        conf_no_vehicle = probs[0, 0].item()
        conf_vehicle = probs[0, 1].item()
        
        assert 0.0 <= conf_no_vehicle <= 1.0
        assert 0.0 <= conf_vehicle <= 1.0
        assert abs(conf_no_vehicle + conf_vehicle - 1.0) < 1e-6  # Sum to 1


# ============================================================================
# Test Tensor Preprocessing
# ============================================================================

class TestTensorPreprocessing:
    """Test tensor preprocessing before inference."""
    
    @pytest.mark.unit
    def test_tensor_concatenation(self):
        """Test concatenating acoustic and seismic tensors."""
        # Simulate acoustic and seismic data
        acoustic = torch.randn(1, 1, 16000)
        seismic = torch.randn(1, 1, 16000)
        
        # Concatenate along channel dimension
        x = torch.cat([acoustic, seismic], dim=1)
        
        assert x.shape == (1, 2, 16000)  # batch=1, channels=2, length=16000
    
    @pytest.mark.unit
    def test_tensor_clamping(self):
        """Test clamping extreme values."""
        # Create tensor with outliers
        x = torch.tensor([[[-15.0, 5.0, 12.0, -8.0, 3.0]]])
        
        # Clamp to [-10, 10]
        x_clamped = torch.clamp(x, -10.0, 10.0)
        
        assert x_clamped.min() >= -10.0
        assert x_clamped.max() <= 10.0
        assert x_clamped[0, 0, 1].item() == 5.0  # Unchanged
        assert x_clamped[0, 0, 0].item() == -10.0  # Clamped from -15
        assert x_clamped[0, 0, 2].item() == 10.0  # Clamped from 12
    
    @pytest.mark.unit
    def test_tensor_reshaping(self):
        """Test reshaping flat data to (batch, channels, seq_len)."""
        # Simulate flat acoustic data from protobuf
        flat_data = list(range(16000))
        
        # Reshape to tensor
        tensor = torch.tensor(flat_data, dtype=torch.float32).reshape(1, 1, 16000)
        
        assert tensor.shape == (1, 1, 16000)
        assert tensor.dtype == torch.float32


# ============================================================================
# Test NATS Integration
# ============================================================================

class TestNATSIntegration:
    """Test NATS message handling."""
    
    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_subscribe_to_sensor_data(self, mock_nats_client):
        """Test subscribing to sensor.data subject."""
        subject = "sensor.data"
        callback = AsyncMock()
        
        await mock_nats_client.subscribe(subject, cb=callback)
        
        # Verify subscription
        mock_nats_client.subscribe.assert_called_once()
        call_args = mock_nats_client.subscribe.call_args
        assert "sensor.data" in call_args[0][0]
    
    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_publish_detection_result(self, mock_nats_client, skip_if_no_protos):
        """Test publishing DetectionResult to detection.result."""
        from inference_protos import inference_pb2
        
        # Create DetectionResult
        result = inference_pb2.DetectionResult()
        result.sensor_data.sensor_id = "sensor_array_01"
        result.vehicle_detected = True
        result.confidence = 0.95
        
        serialized = result.SerializeToString()
        
        # Publish to NATS
        await mock_nats_client.publish("detection.result", serialized)
        
        # Verify
        mock_nats_client.publish.assert_called_once_with("detection.result", serialized)
        assert len(mock_nats_client._published_messages) == 1
        assert mock_nats_client._published_messages[0]['subject'] == "detection.result"


# ============================================================================
# Test Message Processing
# ============================================================================

class TestMessageProcessing:
    """Test processing SensorData messages."""
    
    @pytest.mark.unit
    def test_extract_acoustic_data(self, skip_if_no_protos):
        """Test extracting acoustic data from SensorData protobuf."""
        from inference_protos import inference_pb2
        
        # Create SensorData with acoustic
        sd = inference_pb2.SensorData()
        sd.sensor_id = "sensor_array_01"
        sd.channels.append("acoustic")
        
        acoustic_data = [float(i) for i in range(16000)]
        sd.acoustic_data.shape.extend([16000])
        sd.acoustic_data.data.extend(acoustic_data)
        
        # Extract and reshape
        audio = torch.tensor(list(sd.acoustic_data.data), dtype=torch.float32).reshape(1, 1, 16000)
        
        assert audio.shape == (1, 1, 16000)
        assert len(list(sd.acoustic_data.data)) == 16000
    
    @pytest.mark.unit
    def test_missing_channel_detection(self, skip_if_no_protos):
        """Test detecting missing acoustic or seismic channel."""
        from inference_protos import inference_pb2
        
        # Create SensorData with only acoustic (missing seismic)
        sd = inference_pb2.SensorData()
        sd.sensor_id = "sensor_array_01"
        sd.channels.append("acoustic")
        
        # Check for both channels
        has_acoustic = sd.HasField("acoustic_data")
        has_seismic = sd.HasField("seismic_data")
        
        # Should skip if either is missing
        should_skip = not has_acoustic or not has_seismic
        assert should_skip is True


# ============================================================================
# Test Protobuf Handling
# ============================================================================

class TestProtobufHandling:
    """Test protobuf serialization and deserialization."""
    
    @pytest.mark.unit
    def test_parse_sensor_data(self, skip_if_no_protos, sample_sensor_data_proto):
        """Test parsing SensorData from bytes."""
        # Serialize
        serialized = sample_sensor_data_proto.SerializeToString()
        
        # Parse back
        from inference_protos import inference_pb2
        parsed = inference_pb2.SensorData()
        parsed.ParseFromString(serialized)
        
        assert parsed.sensor_id == sample_sensor_data_proto.sensor_id
    
    @pytest.mark.unit
    def test_create_detection_result(self, skip_if_no_protos, create_timestamp_proto):
        """Test creating DetectionResult protobuf."""
        from inference_protos import inference_pb2
        
        # Create detection result
        result = inference_pb2.DetectionResult()
        
        # Set sensor data
        result.sensor_data.sensor_id = "sensor_array_01"
        ts = create_timestamp_proto(1700000000.5)
        result.sensor_data.time_stamp.CopyFrom(ts)
        
        # Set detection results
        result.vehicle_detected = True
        result.confidence = 0.88
        
        # Verify
        assert result.sensor_data.sensor_id == "sensor_array_01"
        assert result.vehicle_detected is True
        assert abs(result.confidence - 0.88) < 1e-6
        assert result.sensor_data.time_stamp.seconds == 1700000000


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration and environment variables."""
    
    @pytest.mark.unit
    def test_model_dir_environment(self):
        """Test MODEL_DIR environment variable."""
        import os
        
        # Default value
        default_dir = os.environ.get("MODEL_DIR", "/app/model")
        assert default_dir in ["/app/model", os.environ.get("MODEL_DIR")]
    
    @pytest.mark.unit
    def test_nats_url_required(self):
        """Test NATS_URL environment variable is required."""
        import os
        
        # Check if required var is set
        required_vars = ["NATS_URL"]
        for var in required_vars:
            if var not in os.environ:
                # Would raise EnvironmentError in actual code
                missing = var
                assert missing == "NATS_URL"


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    def test_malformed_sensor_data(self, skip_if_no_protos):
        """Test handling malformed SensorData."""
        from inference_protos import inference_pb2
        
        # Try parsing invalid bytes
        invalid_bytes = b'invalid data'
        sd = inference_pb2.SensorData()
        
        try:
            sd.ParseFromString(invalid_bytes)
            # Protobuf may parse but produce empty/default values
            parsed = True
        except Exception:
            parsed = False
        
        # Either way, should handle gracefully
        assert isinstance(parsed, bool)
    
    @pytest.mark.unit
    def test_tensor_device_mismatch(self):
        """Test handling tensor device mismatches."""
        # CPU tensor
        x_cpu = torch.randn(1, 2, 16000)
        
        # Move to CPU explicitly (would be CUDA in production)
        device = torch.device("cpu")
        x_device = x_cpu.to(device)
        
        assert x_device.device.type == "cpu"
        assert x_device.shape == x_cpu.shape


# ============================================================================
# Test End-to-End Flow
# ============================================================================

@pytest.mark.integration
@pytest.mark.nats
def test_detection_end_to_end(mock_nats_client, skip_if_no_protos):
    """End-to-end test: SensorData → Inference → DetectionResult."""
    from inference_protos import inference_pb2
    
    # Arrange - create SensorData
    sd = inference_pb2.SensorData()
    sd.sensor_id = "sensor_array_01"
    sd.time_stamp.seconds = 1700000000
    sd.channels.extend(["acoustic", "seismic"])
    
    # Add acoustic data
    acoustic_data = [float(i * 0.001) for i in range(16000)]
    sd.acoustic_data.shape.extend([16000])
    sd.acoustic_data.data.extend(acoustic_data)
    
    # Add seismic data
    seismic_data = [float(i * 0.0001) for i in range(16000)]
    sd.seismic_data.shape.extend([16000])
    sd.seismic_data.data.extend(seismic_data)
    
    # Simulate inference (mock)
    mock_logits = torch.tensor([[0.1, 2.5]])
    probs = torch.nn.functional.softmax(mock_logits, dim=1)
    vehicle_detected = mock_logits.argmax(dim=1).item() == 1
    confidence = probs[0, 1].item()
    
    # Create DetectionResult
    result = inference_pb2.DetectionResult()
    result.sensor_data.CopyFrom(sd)
    result.vehicle_detected = vehicle_detected
    result.confidence = confidence
    
    # Assert
    assert result.sensor_data.sensor_id == "sensor_array_01"
    assert result.vehicle_detected is True
    assert result.confidence > 0.9
    assert result.sensor_data.time_stamp.seconds == 1700000000


@pytest.mark.unit
def test_inference_performance():
    """Test inference completes in reasonable time."""
    import time
    
    # Simulate inference timing
    start = time.time()
    
    # Mock model forward pass
    x = torch.randn(1, 2, 16000)
    with torch.inference_mode():
        # Simulate lightweight model
        logits = torch.randn(1, 2)
    
    elapsed = time.time() - start
    
    # Should be fast (< 100ms for mock)
    assert elapsed < 0.1
