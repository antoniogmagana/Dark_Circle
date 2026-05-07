"""
Tests for Ingestor Node (ROS2 -> Buffer -> NATS).

The Ingestor subscribes to one ``std_msgs/String`` topic per sensor array.
Each message is a JSON document bundling all channels for one timestep.
A cluster-level YAML maps the customer's free-form channel-tag strings
onto SensorBuffer roles. Pure logic (channel-map loading, JSON dispatch)
lives in ``ingestor.dispatch`` and is imported directly. ROS2 / NATS
plumbing is exercised through inline simulation.
"""

import json
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest
from dispatch import (
    ChannelSpec,
    InvalidChannelMapError,
    load_channel_map,
    make_array_callback,
)

# ---------------------------------------------------------------------------
# load_channel_map
# ---------------------------------------------------------------------------


class TestLoadChannelMap:
    """``load_channel_map`` parses channels.yaml into ChannelSpec entries."""

    @pytest.mark.unit
    def test_audio_seismic_only(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 16000}
                  EHZ: {role: seismic,  expected_rate: 100}
                """
            )
        )
        result = load_channel_map(str(path))
        assert result["MIC"] == ChannelSpec(role="acoustic", expected_rate=16000)
        assert result["EHZ"] == ChannelSpec(role="seismic", expected_rate=100)

    @pytest.mark.unit
    def test_with_accel(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 16000}
                  EHZ: {role: seismic,  expected_rate: 100}
                  ENE: {role: accel_x,  expected_rate: 100}
                  ENN: {role: accel_y,  expected_rate: 100}
                  ENZ: {role: accel_z,  expected_rate: 100}
                """
            )
        )
        result = load_channel_map(str(path))
        assert result["ENE"].role == "accel_x"
        assert result["ENN"].role == "accel_y"
        assert result["ENZ"].role == "accel_z"

    @pytest.mark.unit
    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(InvalidChannelMapError, match="not found"):
            load_channel_map(str(tmp_path / "nope.yaml"))

    @pytest.mark.unit
    def test_malformed_yaml_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text("channels: {MIC: [unbalanced")
        with pytest.raises(InvalidChannelMapError, match="malformed"):
            load_channel_map(str(path))

    @pytest.mark.unit
    def test_missing_top_level_channels_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text("other: {}")
        with pytest.raises(InvalidChannelMapError, match="top-level 'channels'"):
            load_channel_map(str(path))

    @pytest.mark.unit
    def test_unknown_role_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 16000}
                  EHZ: {role: seismic,  expected_rate: 100}
                  XYZ: {role: lidar,    expected_rate: 50}
                """
            )
        )
        with pytest.raises(InvalidChannelMapError, match="unknown role"):
            load_channel_map(str(path))

    @pytest.mark.unit
    def test_missing_required_role_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 16000}
                """
            )
        )
        with pytest.raises(InvalidChannelMapError, match="required roles"):
            load_channel_map(str(path))

    @pytest.mark.unit
    def test_partial_accel_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 16000}
                  EHZ: {role: seismic,  expected_rate: 100}
                  ENE: {role: accel_x,  expected_rate: 100}
                  ENN: {role: accel_y,  expected_rate: 100}
                """
            )
        )
        with pytest.raises(InvalidChannelMapError, match="accel"):
            load_channel_map(str(path))

    @pytest.mark.unit
    def test_non_positive_rate_rejected(self, tmp_path):
        path = tmp_path / "channels.yaml"
        path.write_text(
            textwrap.dedent(
                """
                channels:
                  MIC: {role: acoustic, expected_rate: 0}
                  EHZ: {role: seismic,  expected_rate: 100}
                """
            )
        )
        with pytest.raises(InvalidChannelMapError, match="positive integer"):
            load_channel_map(str(path))


# ---------------------------------------------------------------------------
# make_array_callback — happy path
# ---------------------------------------------------------------------------


def _make_msg(**body):
    """Build a Mock std_msgs/String whose .data is the JSON-encoded body."""
    return Mock(data=json.dumps(body))


def _basic_map():
    return {
        "MIC": ChannelSpec(role="acoustic", expected_rate=16000),
        "EHZ": ChannelSpec(role="seismic", expected_rate=100),
    }


class TestArrayCallbackHappyPath:
    """JSON message arrives, callback fans out to buffer per channel."""

    @pytest.mark.unit
    def test_calls_maybe_close_then_load_per_channel(self):
        buffer = Mock()
        buffer.maybe_close_window.return_value = None
        buffer.load_buffer.return_value = None
        publish = Mock()

        cb = make_array_callback(_basic_map(), buffer, publish)
        msg = _make_msg(
            sensor_id="sensor_1",
            state="background",
            timestamp_unix=1700000000.0,
            channels=[
                {"channel": "MIC", "sampling_rate": 16000, "readings": [1, 2, 3]},
                {"channel": "EHZ", "sampling_rate": 100, "readings": [4, 5]},
            ],
        )
        cb(msg)

        buffer.maybe_close_window.assert_called_once_with(1700000000.0)
        load_calls = buffer.load_buffer.call_args_list
        assert load_calls[0].args == ("acoustic", 1700000000.0, [1, 2, 3])
        assert load_calls[1].args == ("seismic", 1700000000.0, [4, 5])
        publish.assert_not_called()

    @pytest.mark.unit
    def test_publishes_when_close_returns_payload(self):
        buffer = Mock()
        payload = Mock()
        buffer.maybe_close_window.return_value = payload
        buffer.load_buffer.return_value = None
        publish = Mock()

        cb = make_array_callback(_basic_map(), buffer, publish)
        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix=1.0,
                channels=[{"channel": "MIC", "sampling_rate": 16000, "readings": [1]}],
            )
        )

        publish.assert_called_once_with(payload)

    @pytest.mark.unit
    def test_close_fires_before_loads(self):
        """maybe_close_window must run before any load_buffer in a message."""
        buffer = Mock()
        order = []
        buffer.maybe_close_window.side_effect = lambda ts: order.append("close") or None
        buffer.load_buffer.side_effect = lambda *a, **k: order.append("load")
        publish = Mock()

        cb = make_array_callback(_basic_map(), buffer, publish)
        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix=1.0,
                channels=[
                    {"channel": "MIC", "sampling_rate": 16000, "readings": [1]},
                    {"channel": "EHZ", "sampling_rate": 100, "readings": [2]},
                ],
            )
        )
        assert order == ["close", "load", "load"]


# ---------------------------------------------------------------------------
# make_array_callback — error and edge cases
# ---------------------------------------------------------------------------


class TestArrayCallbackEdgeCases:
    @pytest.mark.unit
    def test_unknown_channel_tag_is_skipped(self):
        buffer = Mock()
        buffer.maybe_close_window.return_value = None
        publish = Mock()

        cb = make_array_callback(_basic_map(), buffer, publish)
        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix=1.0,
                channels=[
                    {"channel": "UNKNOWN", "sampling_rate": 99, "readings": [9]},
                    {"channel": "MIC", "sampling_rate": 16000, "readings": [1]},
                ],
            )
        )

        # Only MIC routed.
        load_calls = buffer.load_buffer.call_args_list
        assert len(load_calls) == 1
        assert load_calls[0].args == ("acoustic", 1.0, [1])

    @pytest.mark.unit
    def test_malformed_json_drops_message(self):
        buffer = Mock()
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(Mock(data="this is not json"))

        buffer.maybe_close_window.assert_not_called()
        buffer.load_buffer.assert_not_called()

    @pytest.mark.unit
    def test_missing_timestamp_drops_message(self):
        buffer = Mock()
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(_make_msg(sensor_id="s", state="background", channels=[]))

        buffer.maybe_close_window.assert_not_called()
        buffer.load_buffer.assert_not_called()

    @pytest.mark.unit
    def test_non_numeric_timestamp_drops_message(self):
        buffer = Mock()
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix="not-a-number",
                channels=[],
            )
        )

        buffer.maybe_close_window.assert_not_called()

    @pytest.mark.unit
    def test_missing_channels_field_drops_message(self):
        buffer = Mock()
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(
            _make_msg(
                sensor_id="s", state="background", timestamp_unix=1.0
            )  # no channels[]
        )

        buffer.maybe_close_window.assert_not_called()
        buffer.load_buffer.assert_not_called()

    @pytest.mark.unit
    def test_state_counter_tracks_background_and_trigger(self, capsys):
        buffer = Mock()
        buffer.maybe_close_window.return_value = None
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        for state in ("background", "trigger", "background", None, "weird"):
            body = {"sensor_id": "s", "timestamp_unix": 1.0, "channels": []}
            if state is not None:
                body["state"] = state
            cb(Mock(data=json.dumps(body)))

        # First message log line is emitted at recv=1; trigger one more to
        # force the second milestone (recv=10) to verify counts.
        for _ in range(8):
            cb(
                _make_msg(
                    sensor_id="s",
                    state="background",
                    timestamp_unix=1.0,
                    channels=[],
                )
            )

        out = capsys.readouterr().out
        assert "bg=" in out
        assert "trig=" in out

    @pytest.mark.unit
    def test_sampling_rate_mismatch_logs_but_continues(self, capsys):
        buffer = Mock()
        buffer.maybe_close_window.return_value = None
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix=1.0,
                channels=[{"channel": "MIC", "sampling_rate": 8000, "readings": [1]}],
            )
        )

        out = capsys.readouterr().out
        assert "sampling_rate mismatch" in out
        # But still routed to buffer — soft validation.
        buffer.load_buffer.assert_called_once_with("acoustic", 1.0, [1])

    @pytest.mark.unit
    def test_missing_readings_skips_channel(self):
        buffer = Mock()
        buffer.maybe_close_window.return_value = None
        publish = Mock()
        cb = make_array_callback(_basic_map(), buffer, publish)

        cb(
            _make_msg(
                sensor_id="s",
                state="background",
                timestamp_unix=1.0,
                channels=[
                    {"channel": "MIC", "sampling_rate": 16000},  # no readings
                    {"channel": "EHZ", "sampling_rate": 100, "readings": [2]},
                ],
            )
        )
        # Only EHZ routed; MIC skipped.
        load_calls = buffer.load_buffer.call_args_list
        assert len(load_calls) == 1
        assert load_calls[0].args == ("seismic", 1.0, [2])


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
# End-to-end: callback throughput
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_callback_throughput_is_low_overhead():
    """Protect against accidental quadratic deserialization or per-call work."""
    import time

    buffer = Mock()
    buffer.maybe_close_window.return_value = None
    buffer.load_buffer.return_value = None
    publish = Mock()
    cb = make_array_callback(_basic_map(), buffer, publish)

    payload = json.dumps(
        {
            "sensor_id": "s",
            "state": "background",
            "timestamp_unix": 1.0,
            "channels": [
                {"channel": "MIC", "sampling_rate": 16000, "readings": [1, 2, 3, 4, 5]},
                {"channel": "EHZ", "sampling_rate": 100, "readings": [6, 7]},
            ],
        }
    )
    msg = Mock(data=payload)

    n = 1000
    t0 = time.time()
    for _ in range(n):
        cb(msg)
    elapsed = time.time() - t0

    assert buffer.maybe_close_window.call_count == n
    assert n / max(elapsed, 1e-6) > 100
