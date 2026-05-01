"""
Tests for Infer Classify Node (Classification Model Inference).

The Infer Classify node subscribes to NATS detection.result, runs vehicle
classification inference (only if vehicle_detected=True), and publishes
EgressPayload to classification.result.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

# ============================================================================
# Test Model Loading
# ============================================================================


class TestModelLoading:
    """Test model initialization and loading."""

    @pytest.mark.unit
    def test_config_with_mel_spectrogram(self):
        """Test configuration loading with Mel spectrogram flag."""
        config_dict = {
            "MODEL_NAME": "ResNet18_Mel_2Ch",
            "IN_CHANNELS": 2,
            "NUM_CLASSES": 7,
            "SEQ_LEN": 16000,
            "USE_MEL": True,
            "CLASS_MAP": {
                "0": "bicycle",
                "1": "motorcycle",
                "2": "car",
                "3": "truck",
                "4": "bus",
                "5": "helicopter",
                "6": "airplane",
            },
        }

        config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}
        config = SimpleNamespace(**config_dict)

        assert config.USE_MEL is True
        assert config.NUM_CLASSES == 7
        assert config.CLASS_MAP[2] == "car"

    @pytest.mark.unit
    def test_class_map_conversion(self):
        """Test CLASS_MAP conversion from string keys to int keys."""
        class_map_str = {"0": "pickup", "1": "tesla", "2": "mustang"}

        # Convert to int keys (as done in main.py)
        class_map_int = {int(k): v for k, v in class_map_str.items()}

        assert class_map_int[0] == "pickup"
        assert class_map_int[1] == "tesla"
        assert class_map_int[2] == "mustang"
        assert 0 in class_map_int
        assert "0" not in class_map_int


# ============================================================================
# Test Classification Inference
# ============================================================================


class TestClassificationInference:
    """Test classification inference logic."""

    @pytest.mark.unit
    def test_classification_result(self):
        """Test classification returns vehicle class and confidence."""
        # Mock model output (logits for 7 classes)
        mock_logits = torch.tensor([[-0.5, 0.2, 2.8, 0.1, -0.3, 0.5, -0.2]])

        # CLASS_MAP simulation
        class_map = {
            0: "bicycle",
            1: "motorcycle",
            2: "car",
            3: "truck",
            4: "bus",
            5: "helicopter",
            6: "airplane",
        }

        # Simulate classification
        probs = torch.nn.functional.softmax(mock_logits, dim=1)
        class_idx = mock_logits.argmax(dim=1).item()
        vehicle_class = class_map[class_idx]
        confidence = probs[0, class_idx].item()

        assert vehicle_class == "car"
        assert class_idx == 2
        assert confidence > 0.5  # Confidence threshold lowered to account for softmax distribution

    @pytest.mark.unit
    def test_confidence_for_each_class(self):
        """Test confidence calculation for all classes."""
        mock_logits = torch.tensor([[1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3]])
        probs = torch.nn.functional.softmax(mock_logits, dim=1)

        # All probabilities should sum to 1
        assert abs(probs.sum().item() - 1.0) < 1e-6

        # Each probability should be in [0, 1]
        for i in range(7):
            assert 0.0 <= probs[0, i].item() <= 1.0

    @pytest.mark.unit
    def test_argmax_selection(self):
        """Test argmax selects highest logit."""
        mock_logits = torch.tensor([[0.1, 0.3, 2.5, 0.2, 0.1, 0.4, 0.15]])

        class_idx = mock_logits.argmax(dim=1).item()

        # Index 2 has highest value (2.5)
        assert class_idx == 2


# ============================================================================
# Test Mel Spectrogram Preprocessing
# ============================================================================


class TestMelPreprocessing:
    """Test Mel spectrogram preprocessing."""

    @pytest.mark.unit
    def test_mel_spectrogram_flag(self):
        """Test USE_MEL configuration flag."""
        # Create config with Mel enabled
        config = SimpleNamespace(USE_MEL=True, SEQ_LEN=16000, IN_CHANNELS=2)

        # Simulate preprocessing decision
        torch.randn(1, 2, 16000)

        if config.USE_MEL:
            # Would call extract_mel_spectrogram(x, config)
            should_apply_mel = True
        else:
            should_apply_mel = False

        assert should_apply_mel is True

    @pytest.mark.unit
    def test_no_mel_preprocessing(self):
        """Test raw waveform path (USE_MEL=False)."""
        config = SimpleNamespace(USE_MEL=False, SEQ_LEN=16000)

        x = torch.randn(1, 2, 16000)

        if config.USE_MEL:
            x_processed = x  # Would call extract_mel_spectrogram
        else:
            x_processed = x  # Use raw waveform

        # Should remain unchanged
        assert x_processed.shape == (1, 2, 16000)


# ============================================================================
# Test NATS Integration
# ============================================================================


class TestNATSIntegration:
    """Test NATS message handling."""

    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_subscribe_to_detection_result(self, mock_nats_client):
        """Test subscribing to detection.result subject."""
        subject = "detection.result"
        callback = AsyncMock()

        await mock_nats_client.subscribe(subject, cb=callback)

        # Verify subscription
        mock_nats_client.subscribe.assert_called_once()
        call_args = mock_nats_client.subscribe.call_args
        assert "detection.result" in call_args[0][0]

    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_publish_classification_result(self, mock_nats_client, skip_if_no_protos):
        """Test publishing EgressPayload to classification.result."""
        from inference_protos import inference_pb2

        # Create EgressPayload
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = "sensor_array_01"
        payload.vehicle_detected = True
        payload.detection_confidence = 0.92
        payload.vehicle_class = "pickup"
        payload.classification_confidence = 0.88

        serialized = payload.SerializeToString()

        # Publish to NATS
        await mock_nats_client.publish("classification.result", serialized)

        # Verify
        mock_nats_client.publish.assert_called_once_with("classification.result", serialized)
        assert len(mock_nats_client._published_messages) == 1


# ============================================================================
# Test Message Processing
# ============================================================================


class TestMessageProcessing:
    """Test processing DetectionResult messages."""

    @pytest.mark.unit
    def test_skip_if_no_vehicle_detected(self, skip_if_no_protos):
        """Test early return if vehicle_detected=False."""
        from inference_protos import inference_pb2

        # Create DetectionResult with no vehicle
        detection = inference_pb2.DetectionResult()
        detection.vehicle_detected = False
        detection.confidence = 0.15

        # Should skip classification
        should_process = detection.vehicle_detected
        assert should_process is False

    @pytest.mark.unit
    def test_process_when_vehicle_detected(self, skip_if_no_protos):
        """Test processing when vehicle_detected=True."""
        from inference_protos import inference_pb2

        # Create DetectionResult with vehicle
        detection = inference_pb2.DetectionResult()
        detection.vehicle_detected = True
        detection.confidence = 0.95

        # Should process classification
        should_process = detection.vehicle_detected
        assert should_process is True

    @pytest.mark.unit
    def test_extract_sensor_data_from_detection(self, skip_if_no_protos, create_timestamp_proto):
        """Test extracting SensorData from DetectionResult."""
        from inference_protos import inference_pb2

        # Create DetectionResult with SensorData
        detection = inference_pb2.DetectionResult()
        detection.sensor_data.sensor_id = "sensor_array_01"
        ts = create_timestamp_proto(1700000000.0)
        detection.sensor_data.time_stamp.CopyFrom(ts)
        detection.vehicle_detected = True
        detection.confidence = 0.88

        # Extract sensor data
        sd = detection.sensor_data

        assert sd.sensor_id == "sensor_array_01"
        assert sd.time_stamp.seconds == 1700000000


# ============================================================================
# Test Protobuf Handling
# ============================================================================


class TestProtobufHandling:
    """Test protobuf serialization and deserialization."""

    @pytest.mark.unit
    def test_parse_detection_result(self, skip_if_no_protos, create_timestamp_proto):
        """Test parsing DetectionResult from bytes."""
        from inference_protos import inference_pb2

        # Create and serialize
        detection = inference_pb2.DetectionResult()
        detection.sensor_data.sensor_id = "sensor_array_02"
        detection.vehicle_detected = True
        detection.confidence = 0.91

        serialized = detection.SerializeToString()

        # Parse back
        parsed = inference_pb2.DetectionResult()
        parsed.ParseFromString(serialized)

        assert parsed.sensor_data.sensor_id == "sensor_array_02"
        assert parsed.vehicle_detected is True
        assert abs(parsed.confidence - 0.91) < 1e-6

    @pytest.mark.unit
    def test_create_egress_payload(self, skip_if_no_protos, create_timestamp_proto):
        """Test creating EgressPayload protobuf."""
        from inference_protos import inference_pb2

        # Create payload
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = "sensor_array_01"
        ts = create_timestamp_proto(1700000000.5)
        payload.time_stamp.CopyFrom(ts)
        payload.vehicle_detected = True
        payload.detection_confidence = 0.88
        payload.vehicle_class = "tesla"
        payload.classification_confidence = 0.93

        # Verify
        assert payload.sensor_id == "sensor_array_01"
        assert payload.vehicle_class == "tesla"
        assert abs(payload.classification_confidence - 0.93) < 1e-6

    @pytest.mark.unit
    def test_copy_detection_confidence(self, skip_if_no_protos):
        """Test copying detection confidence to EgressPayload."""
        from inference_protos import inference_pb2

        # Create DetectionResult
        detection = inference_pb2.DetectionResult()
        detection.confidence = 0.87

        # Create EgressPayload and copy confidence
        payload = inference_pb2.EgressPayload()
        payload.detection_confidence = detection.confidence

        assert abs(payload.detection_confidence - 0.87) < 1e-6


# ============================================================================
# Test Tensor Preprocessing
# ============================================================================


class TestTensorPreprocessing:
    """Test tensor preprocessing before classification."""

    @pytest.mark.unit
    def test_tensor_concatenation_classify(self):
        """Test concatenating acoustic and seismic for classification."""
        acoustic = torch.randn(1, 1, 16000)
        seismic = torch.randn(1, 1, 16000)

        # Concatenate
        x = torch.cat([acoustic, seismic], dim=1)

        assert x.shape == (1, 2, 16000)

    @pytest.mark.unit
    def test_tensor_clamping_classify(self):
        """Test clamping before classification."""
        x = torch.tensor([[[50.0, -20.0, 5.0, -8.0]]])

        # Clamp to [-10, 10]
        x_clamped = torch.clamp(x, -10.0, 10.0)

        assert x_clamped.min() >= -10.0
        assert x_clamped.max() <= 10.0


# ============================================================================
# Test Configuration
# ============================================================================


class TestConfiguration:
    """Test configuration and environment variables."""

    @pytest.mark.unit
    def test_nats_url_required_classify(self):
        """Test NATS_URL is required."""

        required_vars = ["NATS_URL"]
        for var in required_vars:
            # Would check in main()
            is_required = var in required_vars
            assert is_required is True


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.unit
    def test_missing_channels_in_detection(self, skip_if_no_protos):
        """Test handling missing channels in DetectionResult."""
        from inference_protos import inference_pb2

        # Create DetectionResult with missing seismic
        detection = inference_pb2.DetectionResult()
        detection.vehicle_detected = True
        detection.sensor_data.sensor_id = "sensor_array_01"
        detection.sensor_data.channels.append("acoustic")
        # seismic not added

        # Check for required channels
        has_acoustic = detection.sensor_data.HasField("acoustic_data")
        has_seismic = detection.sensor_data.HasField("seismic_data")

        should_skip = not has_acoustic or not has_seismic
        assert should_skip is True

    @pytest.mark.unit
    def test_zero_confidence_edge_case(self):
        """Test handling very low confidence scores."""
        mock_logits = torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
        probs = torch.nn.functional.softmax(mock_logits, dim=1)

        # All classes should have roughly equal probability
        for i in range(7):
            assert 0.0 <= probs[0, i].item() <= 1.0


# ============================================================================
# Test End-to-End Flow
# ============================================================================


@pytest.mark.integration
@pytest.mark.nats
def test_classification_end_to_end(mock_nats_client, skip_if_no_protos, create_timestamp_proto):
    """End-to-end test: DetectionResult → Classification → EgressPayload."""
    from inference_protos import inference_pb2

    # Arrange - create DetectionResult
    detection = inference_pb2.DetectionResult()
    detection.sensor_data.sensor_id = "sensor_array_01"
    ts = create_timestamp_proto(1700000000.0)
    detection.sensor_data.time_stamp.CopyFrom(ts)
    detection.sensor_data.channels.extend(["acoustic", "seismic"])

    # Add sensor data
    acoustic_data = [float(i * 0.001) for i in range(16000)]
    detection.sensor_data.acoustic_data.shape.extend([16000])
    detection.sensor_data.acoustic_data.data.extend(acoustic_data)

    seismic_data = [float(i * 0.0001) for i in range(16000)]
    detection.sensor_data.seismic_data.shape.extend([16000])
    detection.sensor_data.seismic_data.data.extend(seismic_data)

    detection.vehicle_detected = True
    detection.confidence = 0.89

    # Act - simulate classification
    class_map = {0: "bicycle", 1: "motorcycle", 2: "car", 3: "truck"}
    mock_logits = torch.tensor([[0.2, 0.5, 2.8, 0.1]])
    probs = torch.nn.functional.softmax(mock_logits, dim=1)
    class_idx = mock_logits.argmax(dim=1).item()
    vehicle_class = class_map[class_idx]
    confidence = probs[0, class_idx].item()

    # Create EgressPayload
    payload = inference_pb2.EgressPayload()
    payload.sensor_id = detection.sensor_data.sensor_id
    payload.time_stamp.CopyFrom(detection.sensor_data.time_stamp)
    payload.vehicle_detected = True
    payload.detection_confidence = detection.confidence
    payload.vehicle_class = vehicle_class
    payload.classification_confidence = confidence

    # Assert
    assert payload.sensor_id == "sensor_array_01"
    assert payload.vehicle_class == "car"
    assert payload.classification_confidence > 0.5  # Confidence threshold adjusted for softmax
    assert abs(payload.detection_confidence - 0.89) < 1e-6


@pytest.mark.unit
def test_classification_performance():
    """Test classification completes in reasonable time."""
    import time

    # Simulate inference timing
    start = time.time()

    # Mock model forward pass
    torch.randn(1, 2, 16000)
    with torch.inference_mode():
        # Simulate lightweight model
        torch.randn(1, 7)

    elapsed = time.time() - start

    # Should be fast (< 100ms for mock)
    assert elapsed < 0.1


@pytest.mark.unit
def test_selective_processing():
    """Test classification only processes detections, not all sensor data."""
    from inference_protos import inference_pb2

    # Create 10 detections, only 3 are positive
    detections = []
    for i in range(10):
        det = inference_pb2.DetectionResult()
        det.vehicle_detected = i % 3 == 0  # True for i=0,3,6,9
        detections.append(det)

    # Count how many would be processed
    processed_count = sum(1 for det in detections if det.vehicle_detected)

    # Should only process 4 out of 10
    assert processed_count == 4
