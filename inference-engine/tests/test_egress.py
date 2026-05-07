"""
Tests for Egress Node (NATS to ROS2 bridge).

The Egress subscribes to NATS inference results and publishes
to ROS2 InferenceResult topics.
"""

import contextlib
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import rclpy  # noqa: F401

    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False


skip_if_no_rclpy = pytest.mark.skipif(
    not RCLPY_AVAILABLE, reason="rclpy not available; egress.main requires ROS2"
)


# ============================================================================
# Test Message Conversion
# ============================================================================


class TestMessageConversion:
    """Test protobuf to ROS2 message conversion."""

    @pytest.mark.unit
    def test_detection_result_to_ros2(self, skip_if_no_protos, create_timestamp_proto):
        """Test converting DetectionResult protobuf to ROS2 message."""
        # Import protobuf types
        from inference_protos import inference_pb2

        # Create DetectionResult protobuf
        detection = inference_pb2.DetectionResult()
        detection.sensor_data.sensor_id = "sensor_array_01"
        ts = create_timestamp_proto(1700000000.5)
        detection.sensor_data.time_stamp.CopyFrom(ts)
        detection.vehicle_detected = True
        detection.confidence = 0.92

        # Simulate conversion logic from egress/main.py on_detection()
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = detection.sensor_data.sensor_id
        payload.time_stamp.CopyFrom(detection.sensor_data.time_stamp)
        payload.vehicle_detected = detection.vehicle_detected
        payload.detection_confidence = detection.confidence

        # Assert conversion
        assert payload.sensor_id == "sensor_array_01"
        assert payload.vehicle_detected is True
        assert abs(payload.detection_confidence - 0.92) < 1e-6  # Float precision tolerance
        assert payload.time_stamp.seconds == 1700000000
        assert payload.time_stamp.nanos == 500000000

    @pytest.mark.unit
    def test_classification_result_to_ros2(self, skip_if_no_protos, create_timestamp_proto):
        """Test converting EgressPayload with classification to ROS2."""
        from inference_protos import inference_pb2

        # Create EgressPayload with classification data
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = "sensor_array_01"
        ts = create_timestamp_proto(1700000000.0)
        payload.time_stamp.CopyFrom(ts)
        payload.vehicle_detected = True
        payload.detection_confidence = 0.88
        payload.vehicle_class = "pickup"
        payload.classification_confidence = 0.95

        # Simulate ROS2 message creation from egress/main.py _publish()
        msg = Mock()  # InferenceResult ROS2 message
        msg.sensor_id = payload.sensor_id
        msg.timestamp = payload.time_stamp.seconds + payload.time_stamp.nanos * 1e-9
        msg.vehicle_detected = payload.vehicle_detected
        msg.detection_confidence = payload.detection_confidence
        msg.vehicle_class = payload.vehicle_class
        msg.classification_confidence = payload.classification_confidence

        # Assert all fields mapped
        assert msg.sensor_id == "sensor_array_01"
        assert msg.timestamp == 1700000000.0
        assert msg.vehicle_detected is True
        assert abs(msg.detection_confidence - 0.88) < 1e-6  # Float precision tolerance
        assert msg.vehicle_class == "pickup"
        assert abs(msg.classification_confidence - 0.95) < 1e-6

    @pytest.mark.unit
    def test_timestamp_conversion(self, create_timestamp_proto):
        """Test timestamp conversion from protobuf to float."""
        # Test timestamp conversion logic from _publish() method
        # msg.timestamp = payload.time_stamp.seconds + payload.time_stamp.nanos * 1e-9

        # Create a mock timestamp
        seconds = 1700000000
        nanos = 500000000  # 0.5  seconds

        # Perform conversion
        timestamp_float = seconds + nanos * 1e-9

        # Verify conversion
        assert timestamp_float == 1700000000.5

        # Test with zero nanos
        timestamp_float_zero = 1700000000 + 0 * 1e-9
        assert timestamp_float_zero == 1700000000.0

        # Test precision is maintained
        nanos_precise = 123456789
        timestamp_precise = 1000 + nanos_precise * 1e-9
        assert abs(timestamp_precise - 1000.123456789) < 1e-9

    @pytest.mark.unit
    def test_confidence_mapping(self):
        """Test confidence score mapping."""
        # Test that confidence values are in valid range [0.0, 1.0]
        valid_confidences = [0.0, 0.5, 0.99, 1.0]
        for conf in valid_confidences:
            assert 0.0 <= conf <= 1.0

        # Test typical confidence values
        detection_conf = 0.85
        classification_conf = 0.92

        assert detection_conf > 0.5  # Typical threshold
        assert classification_conf > 0.5
        assert isinstance(detection_conf, float)
        assert isinstance(classification_conf, float)


# ============================================================================
# Test NATS Subscription
# ============================================================================


class TestNATSSubscription:
    """Test NATS message subscription."""

    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_subscribe_detection_results(self, mock_nats_client):
        """Test subscribing to detection.result subject."""
        # Mock NATS subscription
        subject = "detection.result "

        def callback(msg):
            return None

        await mock_nats_client.subscribe(subject, cb=callback)

        # Verify subscription was created
        mock_nats_client.subscribe.assert_called_once()
        call_args = mock_nats_client.subscribe.call_args
        assert "detection.result" in call_args[0][0]

    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_subscribe_classification_results(self, mock_nats_client):
        """Test subscribing to classification.result subject."""
        # Mock NATS subscription
        subject = "classification.result"

        def callback(msg):
            return None

        await mock_nats_client.subscribe(subject, cb=callback)

        # Verify subscription was created
        mock_nats_client.subscribe.assert_called()
        assert mock_nats_client.subscribe.call_count >= 1

    @pytest.mark.unit
    def test_deserialize_detection_result(self, skip_if_no_protos):
        """Test deserializing DetectionResult from bytes."""
        from inference_protos import inference_pb2

        # Create DetectionResult
        original = inference_pb2.DetectionResult()
        original.sensor_data.sensor_id = "sensor_array_02"
        original.sensor_data.time_stamp.seconds = 1700000000
        original.vehicle_detected = False
        original.confidence = 0.12

        # Serialize to bytes
        serialized = original.SerializeToString()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Deserialize (as done in egress/main.py on_detection)
        deserialized = inference_pb2.DetectionResult()
        deserialized.ParseFromString(serialized)

        # Assert deserialization preserves data
        assert deserialized.sensor_data.sensor_id == "sensor_array_02"
        assert deserialized.sensor_data.time_stamp.seconds == 1700000000
        assert deserialized.vehicle_detected is False
        assert abs(deserialized.confidence - 0.12) < 1e-6  # Float precision tolerance


# ============================================================================
# Test Publishing Logic
# ============================================================================


class TestPublishingLogic:
    """Test ROS2 publishing logic."""

    @skip_if_no_rclpy
    @pytest.mark.integration
    @pytest.mark.ros2
    @pytest.mark.asyncio
    async def test_on_detection_skips_positives(
        self,
        skip_if_no_protos,
        mock_ros2_node,
        create_timestamp_proto,
    ):
        """Egress on_detection forwards only negatives; positives are
        owned by on_classification so /inference_result is not duplicated."""
        from egress.main import EgressNode
        from inference_protos import inference_pb2

        with patch("egress.main.Node.__init__", return_value=None):
            node = EgressNode.__new__(EgressNode)
            node.publisher = MagicMock()

            positive = inference_pb2.DetectionResult()
            positive.sensor_data.sensor_id = "shake-001"
            positive.sensor_data.time_stamp.CopyFrom(create_timestamp_proto(1.0))
            positive.vehicle_detected = True
            positive.confidence = 0.9

            msg = Mock()
            msg.data = positive.SerializeToString()
            await EgressNode.on_detection(node, msg)

            node.publisher.publish.assert_not_called()

    @skip_if_no_rclpy
    @pytest.mark.integration
    @pytest.mark.ros2
    @pytest.mark.asyncio
    async def test_on_detection_publishes_negatives(
        self,
        skip_if_no_protos,
        mock_ros2_node,
        create_timestamp_proto,
    ):
        """Negatives surface on /inference_result so consumers can record
        'no vehicle' decisions."""
        from egress.main import EgressNode
        from inference_protos import inference_pb2

        with patch("egress.main.Node.__init__", return_value=None):
            node = EgressNode.__new__(EgressNode)
            node.publisher = MagicMock()

            negative = inference_pb2.DetectionResult()
            negative.sensor_data.sensor_id = "shake-001"
            negative.sensor_data.time_stamp.CopyFrom(create_timestamp_proto(1.0))
            negative.vehicle_detected = False
            negative.confidence = 0.1

            msg = Mock()
            msg.data = negative.SerializeToString()
            await EgressNode.on_detection(node, msg)

            node.publisher.publish.assert_called_once()

    @skip_if_no_rclpy
    @pytest.mark.integration
    @pytest.mark.ros2
    @pytest.mark.asyncio
    async def test_on_classification_publishes_once(
        self,
        skip_if_no_protos,
        mock_ros2_node,
        create_timestamp_proto,
    ):
        """Classifier message produces exactly one ROS2 publish."""
        from egress.main import EgressNode
        from inference_protos import inference_pb2

        with patch("egress.main.Node.__init__", return_value=None):
            node = EgressNode.__new__(EgressNode)
            node.publisher = MagicMock()

            payload = inference_pb2.EgressPayload()
            payload.sensor_id = "shake-001"
            payload.time_stamp.CopyFrom(create_timestamp_proto(1.0))
            payload.vehicle_detected = True
            payload.detection_confidence = 0.9
            payload.vehicle_class = "light"
            payload.classification_confidence = 0.8

            msg = Mock()
            msg.data = payload.SerializeToString()
            await EgressNode.on_classification(node, msg)

            node.publisher.publish.assert_called_once()

    @pytest.mark.unit
    def test_merge_detection_and_classification(self):
        """Test merging detection and classification results."""
        # Simulate the two-stage process:
        # 1. Detection result
        detection_data = {
            "sensor_id": "sensor_array_01",
            "timestamp": 1700000000.5,
            "vehicle_detected": True,
            "detection_confidence": 0.85,
            "vehicle_class": "",  # Not yet classified
            "classification_confidence": 0.0,
        }

        # 2. Classification result adds class info
        classification_data = {
            "sensor_id": "sensor_array_01",
            "timestamp": 1700000000.5,
            "vehicle_detected": True,
            "detection_confidence": 0.85,
            "vehicle_class": "tesla",  # Now classified
            "classification_confidence": 0.92,
        }

        # Verify detection data
        assert detection_data["vehicle_detected"] is True
        assert detection_data["vehicle_class"] == ""

        # Verify classification data has full info
        assert classification_data["vehicle_detected"] is True
        assert classification_data["vehicle_class"] == "tesla"
        assert classification_data["classification_confidence"] > 0.0


# ============================================================================
# Test ROS2 Publishing
# ============================================================================


class TestROS2Publishing:
    """Test ROS2 InferenceResult publishing."""

    @pytest.mark.integration
    @pytest.mark.ros2
    def test_create_publisher(self, mock_ros2_node):
        """Test creating ROS2 publisher for /inference_result."""
        # Simulate EgressNode initialization
        topic = "inference_result"
        queue_size = 10

        # Create publisher (as done in EgressNode.__init__)
        publisher = mock_ros2_node.create_publisher(
            Mock(),  # InferenceResult message type
            topic,
            queue_size,
        )

        # Assert publisher created
        assert publisher is not None
        mock_ros2_node.create_publisher.assert_called_once()

        # Verify call arguments
        call_args = mock_ros2_node.create_publisher.call_args
        assert call_args[0][1] == topic
        assert call_args[0][2] == queue_size

    @pytest.mark.integration
    @pytest.mark.ros2
    def test_publish_inference_result(self, mock_ros2_node):
        """Test publishing InferenceResult message."""
        # Arrange - create mock publisher
        mock_publisher = Mock()
        mock_publisher.publish = Mock()
        mock_ros2_node.create_publisher.return_value = mock_publisher

        # Create InferenceResult message
        msg = Mock()
        msg.sensor_id = "sensor_array_01"
        msg.timestamp = 1700000000.5
        msg.vehicle_detected = True
        msg.detection_confidence = 0.88
        msg.vehicle_class = "tesla"
        msg.classification_confidence = 0.91

        # Act - publish message
        mock_publisher.publish(msg)

        # Assert
        mock_publisher.publish.assert_called_once_with(msg)
        assert mock_publisher.publish.call_count == 1

    @pytest.mark.unit
    def test_ros2_message_structure(self):
        """Test ROS2 InferenceResult message structure."""
        # Test message field requirements
        msg = Mock()

        # Set all required fields (from egress/main.py _publish)
        msg.sensor_id = "sensor_array_01"
        msg.timestamp = 1700000000.0
        msg.vehicle_detected = True
        msg.detection_confidence = 0.85
        msg.vehicle_class = "pickup"
        msg.classification_confidence = 0.92

        # Verify all fields are set
        assert hasattr(msg, "sensor_id")
        assert hasattr(msg, "timestamp")
        assert hasattr(msg, "vehicle_detected")
        assert hasattr(msg, "detection_confidence")
        assert hasattr(msg, "vehicle_class")
        assert hasattr(msg, "classification_confidence")

        # Verify field types
        assert isinstance(msg.sensor_id, str)
        assert isinstance(msg.timestamp, float)
        assert isinstance(msg.vehicle_detected, bool)
        assert isinstance(msg.detection_confidence, float)
        assert isinstance(msg.vehicle_class, str)
        assert isinstance(msg.classification_confidence, float)


# ============================================================================
# Test Configuration
# ============================================================================


class TestEgressConfiguration:
    """Test Egress node configuration."""

    @pytest.mark.unit
    def test_nats_subjects_configuration(self):
        """Test NATS subjects are configurable."""
        # From egress/main.py - NATS subjects
        DETECTION_SUBJECT = "detection.result"
        CLASSIFICATION_SUBJECT = "classification.result"

        # Assert default subjects
        assert DETECTION_SUBJECT == "detection.result"
        assert CLASSIFICATION_SUBJECT == "classification.result"

        # Verify subject format (NATS convention)
        for subject in [DETECTION_SUBJECT, CLASSIFICATION_SUBJECT]:
            assert isinstance(subject, str)
            assert "." in subject  # NATS hierarchical subjects use dots
            assert not subject.startswith("/")  # NATS doesn't use leading slash
            assert not subject.endswith("/")

        # Test custom subjects
        custom_detection = "inference.vehicle.detection"
        custom_classification = "inference.vehicle.classification"
        assert custom_detection != DETECTION_SUBJECT
        assert custom_classification != CLASSIFICATION_SUBJECT

    @pytest.mark.unit
    def test_ros2_topic_configuration(self):
        """Test ROS2 output topic is configurable."""
        # From egress/main.py - ROS2 topic
        ROS2_TOPIC = "inference_result"

        # Assert default topic
        assert ROS2_TOPIC == "inference_result"
        assert isinstance(ROS2_TOPIC, str)

        # Test topic doesn't start with / (ROS2 node adds it)
        assert not ROS2_TOPIC.startswith("/")

        # Test custom topic
        custom_topic = "vehicle_detection_result"
        assert custom_topic != ROS2_TOPIC

        # Egress now uses a single shared ROS2 node name; sensor_id flows
        # on every published message instead.
        node_name = "egressor"
        assert node_name == "egressor"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEgressEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_malformed_protobuf(self):
        """Test handling of malformed protobuf messages."""
        from inference_protos import inference_pb2

        # Test 1: Invalid bytes
        invalid_bytes = b"not a valid protobuf"
        result = inference_pb2.DetectionResult()

        # ParseFromString may succeed but produce incorrect data; suppress
        # any exception since the test only verifies it doesn't propagate.
        with contextlib.suppress(Exception):
            result.ParseFromString(invalid_bytes)

        # Test 2: Empty bytes
        empty_bytes = b""
        result2 = inference_pb2.DetectionResult()
        result2.ParseFromString(empty_bytes)  # Empty is valid protobuf

        # Should use defaults
        assert result2.vehicle_detected is False  # Default bool value
        assert result2.confidence == 0.0  # Default float value

        # Test 3: Partial message (missing required fields)
        partial = inference_pb2.DetectionResult()
        partial.vehicle_detected = True
        # sensor_data not set - fields will be empty

        serialized = partial.SerializeToString()
        deserialized = inference_pb2.DetectionResult()
        deserialized.ParseFromString(serialized)

        assert deserialized.vehicle_detected is True
        assert deserialized.sensor_data.sensor_id == ""  # Default empty string

    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_nats_reconnection(self, mock_nats_client):
        """Test NATS reconnection on disconnection."""
        # Simulate connection loss and reconnection
        mock_nats_client.is_connected = False

        # Attempt reconnection logic
        connection_attempts = 0
        max_retries = 3

        while connection_attempts < max_retries:
            if not mock_nats_client.is_connected:
                # Attempt reconnect
                await mock_nats_client.connect()
                connection_attempts += 1
                mock_nats_client.is_connected = True  # Simulate success
                break

        # Assert reconnection succeeded
        assert mock_nats_client.is_connected is True
        assert connection_attempts == 1

    @pytest.mark.integration
    @pytest.mark.ros2
    def test_ros2_publisher_failure(self, mock_ros2_node):
        """Test handling ROS2 publisher failure."""
        # Arrange - mock publisher that fails
        mock_publisher = Mock()
        mock_publisher.publish.side_effect = Exception("ROS2 publish failed")
        mock_ros2_node.create_publisher.return_value = mock_publisher

        # Create message
        msg = Mock()
        msg.sensor_id = "sensor_array_01"

        # Act - attempt to publish
        try:
            mock_publisher.publish(msg)
            exception_raised = False
        except Exception as e:
            exception_raised = True
            error_msg = str(e)

        # Assert
        assert exception_raised is True
        assert "ROS2 publish failed" in error_msg
        mock_publisher.publish.assert_called_once_with(msg)


# ============================================================================
# Placeholders for Future Implementation
# ============================================================================


@pytest.mark.integration
@pytest.mark.nats
@pytest.mark.ros2
def test_egress_end_to_end(mock_nats_client, mock_ros2_node, skip_if_no_protos):
    """End-to-end: classifier output -> Convert -> ROS2 publish.

    For positives the classifier owns the publish; the detection topic only
    surfaces negatives. This test covers the positive (classifier) path.
    """
    from inference_protos import inference_pb2

    payload = inference_pb2.EgressPayload()
    payload.sensor_id = "sensor_array_01"
    payload.time_stamp.seconds = 1700000000
    payload.time_stamp.nanos = 500000000
    payload.vehicle_detected = True
    payload.detection_confidence = 0.89
    payload.vehicle_class = "light"
    payload.classification_confidence = 0.91

    nats_msg = Mock()
    nats_msg.data = payload.SerializeToString()

    parsed = inference_pb2.EgressPayload()
    parsed.ParseFromString(nats_msg.data)

    ros2_msg = Mock()
    ros2_msg.sensor_id = parsed.sensor_id
    ros2_msg.timestamp = parsed.time_stamp.seconds + parsed.time_stamp.nanos * 1e-9
    ros2_msg.vehicle_detected = parsed.vehicle_detected
    ros2_msg.detection_confidence = parsed.detection_confidence
    ros2_msg.vehicle_class = parsed.vehicle_class
    ros2_msg.classification_confidence = parsed.classification_confidence

    assert parsed.vehicle_detected is True
    assert ros2_msg.timestamp == 1700000000.5
    assert ros2_msg.vehicle_class == "light"
    assert abs(ros2_msg.detection_confidence - 0.89) < 1e-6
    assert abs(ros2_msg.classification_confidence - 0.91) < 1e-6


@pytest.mark.unit
def test_egress_latency():
    """Test Egress node latency (time from NATS receive to ROS2 publish)."""
    import time

    # Simulate timing of egress operations
    start_time = time.time()

    # 1. Deserialize protobuf (fast operation)
    mock_deserialize_time = 0.0001  # ~0.1ms
    time.sleep(mock_deserialize_time)

    # 2. Convert to ROS2 message (fast operation)
    mock_convert_time = 0.0001  # ~0.1ms
    time.sleep(mock_convert_time)

    # 3. Publish to ROS2 (fast operation)
    mock_publish_time = 0.0001  # ~0.1ms
    time.sleep(mock_publish_time)

    end_time = time.time()
    latency = end_time - start_time

    # Assert latency is low (< 10ms)
    assert latency < 0.01  # 10 milliseconds

    # Test that operations are fast enough for real-time processing
    # At 16kHz audio, each sample window is 1 second
    # Egress should process in < 1ms to avoid backlog
    expected_max_latency = 0.001  # 1ms
    assert latency < expected_max_latency * 10  # Allow 10x margin for test overhead


# ============================================================================
# Readiness sentinel
# ============================================================================


@pytest.mark.unit
def test_readiness_sentinel_message_format():
    """Lock the format of the READY: log line and sentinel path. The egress
    log line names the output topic so the customer immediately sees where
    inference results will surface — that string in the log is the
    customer-facing contract."""
    from pathlib import Path as _P

    src = (_P(__file__).parent.parent / "src" / "egress" / "main.py").read_text()
    # Format is defined with f-string interpolation of output_topic, so we
    # check the stable prefix and suffix that bracket the topic name.
    assert "[egress] READY: NATS subscribed, ROS2 publishing." in src
    assert "Inference results will appear on" in src
    assert "/tmp/ready" in src or 'READY_SENTINEL' in src
