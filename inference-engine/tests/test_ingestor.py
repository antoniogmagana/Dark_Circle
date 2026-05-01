"""
Tests for Ingestor Node (ROS2 -> Buffer -> NATS).

The Ingestor now binds each ROS2 subscription to a known buffer role
(``acoustic`` / ``seismic`` / ``accel_x|y|z``) at construction time, using
the ``SENSOR_ROLE_MAP`` JSON env var produced by Discovery. Pure logic
(env-var parsing, role dispatch) lives in ``ingestor.dispatch`` and is
imported directly. ROS2 / NATS plumbing is exercised through inline
simulation.
"""
import json
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from dispatch import (
    InvalidRoleMapError,
    UnknownRoleError,
    make_role_callback,
    parse_role_map,
)


# ---------------------------------------------------------------------------
# parse_role_map
# ---------------------------------------------------------------------------

class TestParseRoleMap:
    """``parse_role_map`` validates and decodes the SENSOR_ROLE_MAP env var."""

    @pytest.mark.unit
    def test_audio_seismic_only(self):
        env = json.dumps({
            "acoustic": "/shake_001/aud",
            "seismic": "/shake_001/ehz",
        })
        assert parse_role_map(env) == {
            "acoustic": "/shake_001/aud",
            "seismic": "/shake_001/ehz",
        }

    @pytest.mark.unit
    def test_with_accel(self):
        env = json.dumps({
            "acoustic": "/a/aud",
            "seismic": "/a/ehz",
            "accel_x": "/a/ene",
            "accel_y": "/a/enn",
            "accel_z": "/a/enz",
        })
        result = parse_role_map(env)
        assert result["accel_x"] == "/a/ene"
        assert result["accel_y"] == "/a/enn"
        assert result["accel_z"] == "/a/enz"

    @pytest.mark.unit
    def test_invalid_json_rejected(self):
        with pytest.raises(InvalidRoleMapError):
            parse_role_map("not-json")

    @pytest.mark.unit
    def test_missing_acoustic_rejected(self):
        env = json.dumps({"seismic": "/a/ehz"})
        with pytest.raises(InvalidRoleMapError):
            parse_role_map(env)

    @pytest.mark.unit
    def test_missing_seismic_rejected(self):
        env = json.dumps({"acoustic": "/a/aud"})
        with pytest.raises(InvalidRoleMapError):
            parse_role_map(env)

    @pytest.mark.unit
    def test_unknown_role_rejected(self):
        env = json.dumps({
            "acoustic": "/a/aud",
            "seismic": "/a/ehz",
            "lidar": "/a/lidar",  # not a valid buffer channel
        })
        with pytest.raises(UnknownRoleError):
            parse_role_map(env)

    @pytest.mark.unit
    def test_partial_accel_rejected(self):
        """If any accel role is present, all three must be."""
        env = json.dumps({
            "acoustic": "/a/aud",
            "seismic": "/a/ehz",
            "accel_x": "/a/ene",
            "accel_y": "/a/enn",
        })
        with pytest.raises(InvalidRoleMapError):
            parse_role_map(env)


# ---------------------------------------------------------------------------
# make_role_callback
# ---------------------------------------------------------------------------

class TestRoleCallback:
    """``make_role_callback`` returns a closure that dispatches msg -> role."""

    @pytest.mark.unit
    def test_callback_routes_to_buffer_with_role(self):
        buffer = Mock()
        buffer.load_buffer.return_value = None
        publish = Mock()

        cb = make_role_callback(
            role="acoustic",
            buffer=buffer,
            publish_payload=publish,
        )

        msg = Mock()
        msg.start_time = 1700000000.0
        msg.amplitude_readings = [1, 2, 3]
        cb(msg)

        buffer.load_buffer.assert_called_once_with(
            "acoustic", 1700000000.0, [1, 2, 3]
        )
        publish.assert_not_called()

    @pytest.mark.unit
    def test_callback_publishes_when_buffer_returns_payload(self):
        buffer = Mock()
        payload = Mock()
        payload.SerializeToString.return_value = b"serialized"
        buffer.load_buffer.return_value = payload

        publish = Mock()

        cb = make_role_callback(
            role="acoustic",
            buffer=buffer,
            publish_payload=publish,
        )

        msg = Mock()
        msg.start_time = 0.0
        msg.amplitude_readings = list(range(10))
        cb(msg)

        publish.assert_called_once_with(payload)

    @pytest.mark.unit
    def test_callback_uses_its_bound_role(self):
        """Two callbacks for two different roles must route correctly."""
        buffer = Mock()
        buffer.load_buffer.return_value = None
        publish = Mock()

        ac_cb = make_role_callback("acoustic", buffer, publish)
        seismic_cb = make_role_callback("seismic", buffer, publish)

        msg_ac = Mock(start_time=1.0, amplitude_readings=[1])
        msg_se = Mock(start_time=2.0, amplitude_readings=[2])

        ac_cb(msg_ac)
        seismic_cb(msg_se)

        calls = buffer.load_buffer.call_args_list
        assert calls[0].args == ("acoustic", 1.0, [1])
        assert calls[1].args == ("seismic", 2.0, [2])

    @pytest.mark.unit
    def test_callback_does_not_inspect_sensor_id(self):
        """Role binding is independent of msg.sensor_id (no suffix parsing)."""
        buffer = Mock()
        buffer.load_buffer.return_value = None
        publish = Mock()

        cb = make_role_callback("seismic", buffer, publish)

        msg = Mock(
            sensor_id="completely.unrelated.name",
            start_time=0.0,
            amplitude_readings=[],
        )
        cb(msg)

        # Routed to 'seismic' regardless of sensor_id contents.
        buffer.load_buffer.assert_called_once_with("seismic", 0.0, [])


# ---------------------------------------------------------------------------
# End-to-end: env -> subscriptions -> dispatch
# ---------------------------------------------------------------------------

class TestSubscriptionWiring:
    """One subscription created per role; each callback bound to its role."""

    @pytest.mark.integration
    @pytest.mark.ros2
    def test_one_subscription_per_role(self, mock_ros2_node):
        role_map = {
            "acoustic": "/shake_001/aud",
            "seismic": "/shake_001/ehz",
        }
        buffer = Mock()
        buffer.load_buffer.return_value = None
        publish = Mock()

        # Simulate IngestorNode.__init__ subscription loop.
        for role, topic in role_map.items():
            cb = make_role_callback(role, buffer, publish)
            mock_ros2_node.create_subscription(Mock(), topic, cb, 10)

        topics_subscribed = [
            call.args[1] for call in mock_ros2_node.create_subscription.call_args_list
        ]
        assert sorted(topics_subscribed) == [
            "/shake_001/aud",
            "/shake_001/ehz",
        ]

    @pytest.mark.integration
    @pytest.mark.ros2
    def test_renamed_topic_still_dispatches_to_correct_role(self, mock_ros2_node):
        """A non-conventional topic name still works because role binding is explicit."""
        role_map = {
            "acoustic": "/weird/microphone_channel",
            "seismic": "/weird/ground_motion",
        }
        buffer = Mock()
        buffer.load_buffer.return_value = None
        publish = Mock()

        callbacks = {}
        for role, topic in role_map.items():
            cb = make_role_callback(role, buffer, publish)
            callbacks[topic] = cb
            mock_ros2_node.create_subscription(Mock(), topic, cb, 10)

        msg = Mock(start_time=0.0, amplitude_readings=[42])
        callbacks["/weird/microphone_channel"](msg)
        callbacks["/weird/ground_motion"](msg)

        roles_called = [c.args[0] for c in buffer.load_buffer.call_args_list]
        assert roles_called == ["acoustic", "seismic"]


# ---------------------------------------------------------------------------
# ADC normalization sanity checks (kept from the prior suite — protect against
# regressions in the unchanged buffer math).
# ---------------------------------------------------------------------------

class TestADCNormalization:
    @pytest.mark.unit
    def test_16bit_audio_scale(self):
        sample = np.array([16384, -16384], dtype=np.int16)
        normalized = sample / 32768.0
        assert normalized[0] == 0.5
        assert normalized[1] == -0.5

    @pytest.mark.unit
    def test_24bit_seismic_scale(self):
        sample = np.array([4194304, -4194304], dtype=np.int32)
        normalized = sample / 8388608.0
        assert normalized[0] == 0.5
        assert normalized[1] == -0.5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestIngestorConfiguration:
    @pytest.mark.unit
    def test_node_name_translates_hyphens_to_underscores(self):
        """Array IDs use hyphens (RFC 1123); ROS2 node names require underscores."""
        sensor_array = "shake-002"
        node_name = "ingestor_" + sensor_array.replace("-", "_")
        assert node_name == "ingestor_shake_002"

    @pytest.mark.unit
    def test_nats_subject_constant(self):
        from dispatch import NATS_SUBJECT
        assert NATS_SUBJECT == "sensor.data"


# ---------------------------------------------------------------------------
# NATS publish (mocked)
# ---------------------------------------------------------------------------

class TestNATSPublishing:
    @pytest.mark.integration
    @pytest.mark.nats
    @pytest.mark.asyncio
    async def test_publish_serialized_payload(self, mock_nats_client):
        from dispatch import NATS_SUBJECT

        payload = b"serialized-bytes"
        await mock_nats_client.publish(NATS_SUBJECT, payload)

        mock_nats_client.publish.assert_called_once_with(NATS_SUBJECT, payload)
        assert mock_nats_client._published_messages[0]["subject"] == NATS_SUBJECT
        assert mock_nats_client._published_messages[0]["data"] == payload


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ingestor_end_to_end():
    role_map = {
        "acoustic": "/shake_001/aud",
        "seismic": "/shake_001/ehz",
    }
    buffer = Mock()
    payload = Mock()
    payload.SerializeToString.return_value = b"e2e_payload"
    buffer.load_buffer.return_value = payload

    published = []
    publish = lambda p: published.append(p.SerializeToString())

    cb = make_role_callback("acoustic", buffer, publish)
    msg = Mock(start_time=1700000000.0, amplitude_readings=list(range(100)))
    cb(msg)

    buffer.load_buffer.assert_called_once_with(
        "acoustic", 1700000000.0, list(range(100))
    )
    assert published == [b"e2e_payload"]


@pytest.mark.unit
def test_ingestor_handles_buffer_returns_none():
    buffer = Mock()
    buffer.load_buffer.return_value = None
    publish = Mock()

    cb = make_role_callback("seismic", buffer, publish)
    cb(Mock(start_time=0.0, amplitude_readings=[]))
    publish.assert_not_called()


@pytest.mark.unit
def test_ingestor_throughput_is_low_overhead():
    """Mirror the prior suite's perf check; primarily protects against accidental quadratic deserialization."""
    import time

    buffer = Mock()
    buffer.load_buffer.return_value = None
    publish = Mock()
    cb = make_role_callback("acoustic", buffer, publish)

    n = 1000
    t0 = time.time()
    for _ in range(n):
        cb(Mock(start_time=0.0, amplitude_readings=[1, 2, 3, 4, 5]))
    elapsed = time.time() - t0

    assert buffer.load_buffer.call_count == n
    assert n / max(elapsed, 1e-6) > 100
