"""
Comprehensive tests for SensorBuffer class.

The buffer's public API (used by the dispatch callback once per JSON message):

    payload = buffer.maybe_close_window(timestamp_unix)
    if payload is not None: publish_payload(payload)
    for entry in channels:
        buffer.load_buffer(role, timestamp_unix, readings)

Tests cover:
- Initialization and configuration
- Window-close decision (maybe_close_window)
- Per-channel data loading and straddle (load_buffer)
- Holding pen logic for samples that straddle a window
- Rate-mismatch validation at window close
- ADC normalization
- Protobuf packaging
- Edge cases (early data, far-future, empty arrays)
"""

import numpy as np
import pytest
from ingestor.buffer import SensorBuffer


def _close_then_load(buffer, channel, timestamp, data):
    """Mirror the dispatch callback's per-message contract.

    Production calls ``maybe_close_window`` once per JSON message before
    fanning out per-channel; tests that exercise window-boundary behavior
    must do the same. Returns whatever payload (if any) the close fired.
    """
    payload = buffer.maybe_close_window(timestamp)
    buffer.load_buffer(channel, timestamp, data)
    return payload


# ============================================================================
# Test Initialization
# ============================================================================


class TestSensorBufferInit:
    """Test SensorBuffer initialization and configuration."""

    @pytest.mark.unit
    def test_init_basic(self, sensor_id):
        """Test basic initialization."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.sensor_id == sensor_id
        assert buffer.start_time is None
        assert buffer.window == 1.0
        assert len(buffer.active_channels) == 0

    @pytest.mark.unit
    def test_init_default_sample_rates(self, sensor_id):
        """No expected_rates argument falls back to legacy defaults."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.rates["acoustic"] == 16000
        assert buffer.rates["seismic"] == 100
        assert buffer.rates["accel_x"] == 100
        assert buffer.rates["accel_y"] == 100
        assert buffer.rates["accel_z"] == 100

    @pytest.mark.unit
    def test_init_expected_rates_override(self, sensor_id):
        """expected_rates from the channel map overrides the defaults."""
        buffer = SensorBuffer(
            sensor_id,
            expected_rates={"acoustic": 8000, "seismic": 50},
        )
        assert buffer.rates["acoustic"] == 8000
        assert buffer.rates["seismic"] == 50

    @pytest.mark.unit
    def test_init_buffer_sizes(self, sensor_id):
        """Test buffer sizes match sample rates."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.limits["acoustic"] == 16000
        assert buffer.limits["seismic"] == 100
        assert buffer.limits["accel_x"] == 100
        assert buffer.limits["accel_y"] == 100
        assert buffer.limits["accel_z"] == 100

        assert buffer.buffers["acoustic"].shape == (16000,)
        assert buffer.buffers["seismic"].shape == (100,)

    @pytest.mark.unit
    def test_init_adc_scales(self, sensor_id):
        """Test ADC scales for normalization."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.adc_scale["acoustic"] == 2**15  # 16-bit
        assert buffer.adc_scale["seismic"] == 2**23  # 24-bit
        assert buffer.adc_scale["accel"] == 2**23  # 24-bit

    @pytest.mark.unit
    def test_init_holding_pen(self, sensor_id):
        """Test holding pen is initialized empty."""
        buffer = SensorBuffer(sensor_id)

        for channel in ["acoustic", "seismic", "accel_x", "accel_y", "accel_z"]:
            assert buffer.holding_pen[channel] == []

    @pytest.mark.unit
    def test_init_received_samples_zero(self, sensor_id):
        buffer = SensorBuffer(sensor_id)
        for ch in ("acoustic", "seismic", "accel_x", "accel_y", "accel_z"):
            assert buffer.received_samples[ch] == 0


# ============================================================================
# Test maybe_close_window
# ============================================================================


class TestMaybeCloseWindow:
    @pytest.mark.unit
    def test_first_message_initializes_start_time(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        result = buffer.maybe_close_window(base_timestamp)
        assert result is None
        assert buffer.start_time == base_timestamp

    @pytest.mark.unit
    def test_within_window_returns_none(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        result = buffer.maybe_close_window(base_timestamp + 0.5)
        assert result is None
        assert buffer.start_time == base_timestamp

    @pytest.mark.unit
    def test_silent_stream_guard_snaps_start_time(self, sensor_id, base_timestamp):
        """Far-future timestamp on a buffer with no data must not phantom-publish."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # No data loaded; later message arrives 5s later.
        result = buffer.maybe_close_window(base_timestamp + 5.0)
        assert result is None
        assert buffer.start_time == base_timestamp + 5.0

    @pytest.mark.unit
    def test_close_returns_payload_when_active(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        buffer = SensorBuffer(sensor_id)
        # Fill exactly one full window of acoustic at 16 kHz.
        _close_then_load(
            buffer, "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        # Next message timestamp crosses the boundary.
        result = buffer.maybe_close_window(base_timestamp + 1.0)
        assert result is not None
        # start_time advanced.
        assert buffer.start_time == base_timestamp + 1.0


# ============================================================================
# Test Basic Loading
# ============================================================================


class TestBasicLoading:
    @pytest.mark.unit
    def test_load_first_data(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        result = buffer.load_buffer("acoustic", base_timestamp, np.array([100, 200, 300]))

        assert buffer.start_time == base_timestamp
        assert result is None
        assert "acoustic" in buffer.active_channels

    @pytest.mark.unit
    def test_load_early_data_rejected(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp
        # Data 0.1s before window start
        result = buffer.load_buffer(
            "acoustic", base_timestamp - 0.1, np.array([1, 2, 3])
        )
        assert result is None
        assert "acoustic" not in buffer.active_channels

    @pytest.mark.unit
    def test_load_channels_registered(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer("acoustic", base_timestamp, np.array([1, 2, 3]))
        assert "acoustic" in buffer.active_channels

        buffer.load_buffer("seismic", base_timestamp, np.array([4, 5]))
        assert "seismic" in buffer.active_channels
        assert len(buffer.active_channels) == 2

    @pytest.mark.unit
    def test_load_data_at_correct_position(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # 100 samples at t=0 → index 0
        buffer.load_buffer("acoustic", base_timestamp, np.full(100, 42.0))
        assert np.all(buffer.buffers["acoustic"][0:100] == 42.0)
        assert np.all(buffer.buffers["acoustic"][100:200] == 0.0)

        # 100 samples at t=0.1 → index 1600
        buffer.load_buffer("acoustic", base_timestamp + 0.1, np.full(100, 99.0))
        assert np.all(buffer.buffers["acoustic"][1600:1700] == 99.0)


# ============================================================================
# Test Window Boundary Handling
# ============================================================================


class TestWindowBoundaries:
    @pytest.mark.unit
    def test_close_then_load_packages_window(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        """Filling a window's worth of acoustic, then a new-window timestamp,
        must produce a payload from maybe_close_window."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # Fill exactly one window.
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )

        # New message at next-window timestamp.
        result = buffer.maybe_close_window(base_timestamp + 1.0)
        assert result is not None
        assert buffer.start_time == base_timestamp + 1.0

    @pytest.mark.unit
    def test_straddle_chunk2_lands_in_pen(self, sensor_id, base_timestamp):
        """A single message that straddles the boundary keeps chunk_1 in the
        current window and stashes chunk_2 in the holding pen."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # 99 samples at t=0.99 → start_index 99, leaves 1 slot, 9 carry.
        buffer.load_buffer("seismic", base_timestamp + 0.99, np.full(10, 50.0))

        assert buffer.buffers["seismic"][99] == 50.0
        assert len(buffer.holding_pen["seismic"]) == 9

    @pytest.mark.unit
    def test_holding_pen_transferred_on_reset(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        """After window close + reset, pen samples land at index 0 of next window."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # Fill acoustic so the close has something to package.
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )

        # Pre-fill seismic up to t=0.99 so the rate guard accepts the window.
        # Then a final straddling chunk at t=0.99 takes 1 sample into the
        # current window and pushes 9 into the holding pen.
        buffer.load_buffer(
            "seismic", base_timestamp, np.full(99, 0.0)
        )  # 99 samples landing at indices 0..98
        buffer.load_buffer("seismic", base_timestamp + 0.99, np.full(10, 99.0))
        # 1 sample at index 99 (chunk_1), 9 samples in pen (chunk_2).
        assert buffer.buffers["seismic"][99] == 99.0
        assert len(buffer.holding_pen["seismic"]) == 9
        assert buffer.received_samples["seismic"] == 100  # 99 + 1

        # New message at next-window ts triggers close + reset.
        payload = buffer.maybe_close_window(base_timestamp + 1.0)
        assert payload is not None

        # Pen drained into next window's buffer.
        assert len(buffer.holding_pen["seismic"]) == 0
        assert buffer.buffers["seismic"][0] == 99.0
        assert "seismic" in buffer.active_channels  # re-registered
        assert buffer.received_samples["seismic"] == 9  # carried-over count


# ============================================================================
# Test Multi-Channel Synchronization
# ============================================================================


class TestMultiChannel:
    @pytest.mark.unit
    def test_multiple_channels_same_window(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer("acoustic", base_timestamp, np.full(1000, 1.0))
        buffer.load_buffer("seismic", base_timestamp + 0.1, np.full(10, 2.0))
        buffer.load_buffer("accel_x", base_timestamp + 0.2, np.full(5, 3.0))

        assert len(buffer.active_channels) == 3
        assert buffer.buffers["acoustic"][0] == 1.0
        assert buffer.buffers["seismic"][10] == 2.0
        assert buffer.buffers["accel_x"][20] == 3.0

    @pytest.mark.unit
    def test_channels_with_different_rates(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer("acoustic", base_timestamp, np.array([100.0]))
        buffer.load_buffer("seismic", base_timestamp, np.array([200.0]))

        buffer.load_buffer("acoustic", base_timestamp + 0.5, np.array([101.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.5, np.array([201.0]))

        assert buffer.buffers["acoustic"][0] == 100.0
        assert buffer.buffers["acoustic"][8000] == 101.0
        assert buffer.buffers["seismic"][0] == 200.0
        assert buffer.buffers["seismic"][50] == 201.0

    @pytest.mark.unit
    def test_accelerometer_three_axes(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer("accel_x", base_timestamp, np.full(10, 1.0))
        buffer.load_buffer("accel_y", base_timestamp, np.full(10, 2.0))
        buffer.load_buffer("accel_z", base_timestamp, np.full(10, 3.0))

        assert "accel_x" in buffer.active_channels
        assert "accel_y" in buffer.active_channels
        assert "accel_z" in buffer.active_channels
        assert buffer.buffers["accel_x"][0] == 1.0
        assert buffer.buffers["accel_y"][0] == 2.0
        assert buffer.buffers["accel_z"][0] == 3.0


# ============================================================================
# Test Window-Close Rate Validation
# ============================================================================


class TestWindowDropOnRateMismatch:
    """Drift outside ±RATE_TOLERANCE drops the window with a log line."""

    @pytest.mark.unit
    def test_perfect_rate_passes(self, sensor_id, base_timestamp, skip_if_no_protos):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        # Exactly limits[acoustic] samples → drift 0.
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )
        # And one expected sample for seismic so the seismic check is triggered.
        buffer.load_buffer(
            "seismic", base_timestamp, np.full(buffer.limits["seismic"], 1.0)
        )
        payload = buffer.maybe_close_window(base_timestamp + 1.0)
        assert payload is not None

    @pytest.mark.unit
    def test_undercount_drops_window(self, sensor_id, base_timestamp, capsys):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        # Half the expected acoustic samples → drift 0.5, well outside tolerance.
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"] // 2, 1.0)
        )
        result = buffer.maybe_close_window(base_timestamp + 1.0)
        assert result is None
        out = capsys.readouterr().out
        assert "rate mismatch" in out

    @pytest.mark.unit
    def test_within_tolerance_passes(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        # 0.5% drift on acoustic; under 1% tolerance.
        n = buffer.limits["acoustic"] - int(buffer.limits["acoustic"] * 0.005)
        buffer.load_buffer("acoustic", base_timestamp, np.full(n, 1.0))
        buffer.load_buffer(
            "seismic", base_timestamp, np.full(buffer.limits["seismic"], 1.0)
        )
        payload = buffer.maybe_close_window(base_timestamp + 1.0)
        assert payload is not None

    @pytest.mark.unit
    def test_received_samples_resets_after_close(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.full(buffer.limits["seismic"], 1.0)
        )
        buffer.maybe_close_window(base_timestamp + 1.0)

        # After close, counters zeroed for next window (no pen carry here).
        assert buffer.received_samples["acoustic"] == 0
        assert buffer.received_samples["seismic"] == 0


# ============================================================================
# Test Package Output (ADC normalization modes)
# ============================================================================


class TestPackageOutput:
    """Per-window output contract emitted by ``_package_window``.

    Tests drive ``_package_window`` directly by populating ``buffer.buffers``,
    ``buffer.active_channels``, and ``buffer.received_samples`` so the
    assertion is on the package math, not on the streaming loader.
    """

    def _ramp(self, n, lo, hi):
        return np.linspace(lo, hi, n).astype(np.float64)

    def _force_active(self, buffer, *channels):
        """Mark channels active and pre-set their received_samples to the
        full-window count so the rate-mismatch guard accepts the package."""
        for ch in channels:
            buffer.active_channels.add(ch)
            buffer.received_samples[ch] = buffer.limits[ch]

    @pytest.mark.unit
    def test_acoustic_default_preserves_raw_counts(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", False)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["acoustic"], -32768, 32767)
        buffer.buffers["acoustic"] = raw.copy()
        self._force_active(buffer, "acoustic")

        payload = buffer._package_window()
        out = np.array(payload.acoustic_data.data)

        assert abs(out.mean()) < 1e-6
        np.testing.assert_allclose(out, raw - raw.mean(), atol=1e-3)
        assert out.std() > 1000

    @pytest.mark.unit
    def test_acoustic_legacy_normalizes_to_unit_range(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", True)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["acoustic"], -32768, 32767)
        buffer.buffers["acoustic"] = raw.copy()
        self._force_active(buffer, "acoustic")

        payload = buffer._package_window()
        out = np.array(payload.acoustic_data.data)

        scaled = raw / (2**15)
        np.testing.assert_allclose(out, scaled - scaled.mean(), atol=1e-7)
        assert abs(out.max()) < 1.01 and abs(out.min()) < 1.01

    @pytest.mark.unit
    def test_seismic_default_preserves_raw_counts(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", False)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["seismic"], -8388608, 8388607)
        buffer.buffers["seismic"] = raw.copy()
        self._force_active(buffer, "seismic")

        payload = buffer._package_window()
        out = np.array(payload.seismic_data.data)

        assert abs(out.mean()) < 1e-3
        np.testing.assert_allclose(out, raw - raw.mean(), atol=1e-2)
        assert out.std() > 1e5

    @pytest.mark.unit
    def test_seismic_legacy_normalizes_to_unit_range(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", True)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["seismic"], -8388608, 8388607)
        buffer.buffers["seismic"] = raw.copy()
        self._force_active(buffer, "seismic")

        payload = buffer._package_window()
        out = np.array(payload.seismic_data.data)

        scaled = raw / (2**23)
        np.testing.assert_allclose(out, scaled - scaled.mean(), atol=1e-7)
        assert abs(out.max()) < 1.01 and abs(out.min()) < 1.01

    @pytest.mark.unit
    def test_accel_default_preserves_raw_counts(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", False)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        n = buffer.limits["accel_x"]
        buffer.buffers["accel_x"] = np.full(n, 8388607.0)
        buffer.buffers["accel_y"] = np.full(n, -8388608.0)
        buffer.buffers["accel_z"] = np.zeros(n)
        self._force_active(buffer, "accel_x", "accel_y", "accel_z")

        payload = buffer._package_window()
        accel = np.array(payload.accel_data.data).reshape((3, n))
        np.testing.assert_allclose(accel, 0.0, atol=1e-3)

    @pytest.mark.unit
    def test_accel_legacy_normalizes_to_unit_range(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", True)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        n = buffer.limits["accel_x"]
        ramp = self._ramp(n, -8388608, 8388607)
        buffer.buffers["accel_x"] = ramp.copy()
        buffer.buffers["accel_y"] = ramp.copy()
        buffer.buffers["accel_z"] = ramp.copy()
        self._force_active(buffer, "accel_x", "accel_y", "accel_z")

        payload = buffer._package_window()
        accel = np.array(payload.accel_data.data).reshape((3, n))
        scaled = ramp / (2**23)
        expected = scaled - scaled.mean()
        np.testing.assert_allclose(accel[0], expected, atol=1e-7)
        np.testing.assert_allclose(accel[1], expected, atol=1e-7)
        np.testing.assert_allclose(accel[2], expected, atol=1e-7)
        assert abs(accel.max()) < 1.01


# ============================================================================
# Test Protobuf Packaging — full pipeline
# ============================================================================


class TestProtobufPackaging:
    @pytest.mark.unit
    def test_package_sensor_id(self, sensor_id, base_timestamp, skip_if_no_protos):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.zeros(buffer.limits["seismic"])
        )
        payload = buffer.maybe_close_window(base_timestamp + 1.0)

        assert payload is not None
        assert payload.sensor_id == sensor_id

    @pytest.mark.unit
    def test_package_timestamp(self, sensor_id, base_timestamp, skip_if_no_protos):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.zeros(buffer.limits["seismic"])
        )
        payload = buffer.maybe_close_window(base_timestamp + 1.0)

        assert payload is not None
        assert payload.time_stamp.seconds == int(base_timestamp)
        assert payload.time_stamp.nanos == int((base_timestamp - int(base_timestamp)) * 1e9)

    @pytest.mark.unit
    def test_package_channels_list(self, sensor_id, base_timestamp, skip_if_no_protos):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.zeros(buffer.limits["seismic"])
        )
        buffer.load_buffer(
            "accel_x", base_timestamp, np.zeros(buffer.limits["accel_x"])
        )
        buffer.load_buffer(
            "accel_y", base_timestamp, np.zeros(buffer.limits["accel_y"])
        )
        buffer.load_buffer(
            "accel_z", base_timestamp, np.zeros(buffer.limits["accel_z"])
        )

        payload = buffer.maybe_close_window(base_timestamp + 1.0)

        assert payload is not None
        assert "acoustic" in payload.channels
        assert "seismic" in payload.channels
        assert "accel" in payload.channels

    @pytest.mark.unit
    def test_package_tensor_shapes(self, sensor_id, base_timestamp, skip_if_no_protos):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.zeros(buffer.limits["seismic"])
        )
        buffer.load_buffer(
            "accel_x", base_timestamp, np.zeros(buffer.limits["accel_x"])
        )
        buffer.load_buffer(
            "accel_y", base_timestamp, np.zeros(buffer.limits["accel_y"])
        )
        buffer.load_buffer(
            "accel_z", base_timestamp, np.zeros(buffer.limits["accel_z"])
        )

        payload = buffer.maybe_close_window(base_timestamp + 1.0)

        assert payload is not None
        assert payload.acoustic_data.shape == [16000]
        assert payload.seismic_data.shape == [100]
        assert payload.accel_data.shape == [3, 100]

    @pytest.mark.unit
    def test_package_only_active_channels(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )
        payload = buffer.maybe_close_window(base_timestamp + 1.0)

        assert payload is not None
        assert "acoustic" in payload.channels
        assert "seismic" not in payload.channels
        assert "accel" not in payload.channels


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    @pytest.mark.unit
    def test_data_far_future_packages_active_window(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        """A long gap with active data: maybe_close_window packages and resets,
        then the next message's data lands in the new window."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )

        # 2-second gap on the next message.
        result = buffer.maybe_close_window(base_timestamp + 2.0)
        assert result is not None
        # start_time advanced by exactly one window (reset bumps by self.window).
        assert buffer.start_time == base_timestamp + 1.0

    @pytest.mark.unit
    def test_data_far_future_after_silence_does_not_phantom_publish(
        self, sensor_id, base_timestamp
    ):
        """Silent stream guard: no data, far-future ts → snap, no payload."""
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        result = buffer.maybe_close_window(base_timestamp + 5.0)
        assert result is None
        assert buffer.start_time == base_timestamp + 5.0

    @pytest.mark.unit
    def test_single_sample(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        buffer.load_buffer("acoustic", base_timestamp, np.array([42.0]))
        assert buffer.buffers["acoustic"][0] == 42.0

    @pytest.mark.unit
    def test_empty_array(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        result = buffer.load_buffer("acoustic", base_timestamp, np.array([]))
        assert result is None

    @pytest.mark.unit
    def test_reset_clears_active_channels(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)
        buffer.load_buffer(
            "acoustic", base_timestamp, np.full(buffer.limits["acoustic"], 1.0)
        )
        buffer.load_buffer(
            "seismic", base_timestamp, np.full(buffer.limits["seismic"], 1.0)
        )
        assert len(buffer.active_channels) == 2

        # New-window timestamp → close + reset.
        buffer.maybe_close_window(base_timestamp + 1.0)

        # active_channels cleared (no pen carry, so no re-registration).
        assert len(buffer.active_channels) == 0


# ============================================================================
# Test Complex Scenarios
# ============================================================================


class TestComplexScenarios:
    @pytest.mark.unit
    def test_realistic_acoustic_chunks(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        """Many small chunks across a window, then a new-window timestamp closes it."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # 160 chunks of 100 samples = exactly one acoustic window.
        for i in range(160):
            ts = base_timestamp + (i * 100.0 / 16000.0)
            buffer.load_buffer("acoustic", ts, np.full(100, i, dtype=np.float64))

        payload = buffer.maybe_close_window(base_timestamp + 1.0)
        assert payload is not None

    @pytest.mark.unit
    def test_interleaved_multi_channel(self, sensor_id, base_timestamp):
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        for t in np.arange(0, 0.9, 0.01):
            ts = base_timestamp + t
            buffer.load_buffer("acoustic", ts, np.random.randn(160))
            buffer.load_buffer("seismic", ts, np.array([t]))
            buffer.load_buffer("accel_x", ts, np.array([t]))
            buffer.load_buffer("accel_y", ts, np.array([t * 2]))
            buffer.load_buffer("accel_z", ts, np.array([t * 3]))

        assert len(buffer.active_channels) == 5

    @pytest.mark.unit
    def test_sparse_seismic_arrival(
        self, sensor_id, base_timestamp, skip_if_no_protos
    ):
        """Sparse seismic samples within a window are preserved at correct positions."""
        buffer = SensorBuffer(sensor_id)
        buffer.maybe_close_window(base_timestamp)

        # Fill exactly one acoustic window so the rate guard passes.
        buffer.load_buffer(
            "acoustic", base_timestamp, np.zeros(buffer.limits["acoustic"])
        )

        # Sparse seismic: only 3 of the expected 100 samples. Drift > 1% so
        # the rate guard should drop this window. We assert that drop
        # rather than the sparse-sample positions, since the guard fires first.
        buffer.load_buffer("seismic", base_timestamp + 0.0, np.array([1.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.5, np.array([2.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.9, np.array([3.0]))

        payload = buffer.maybe_close_window(base_timestamp + 1.0)
        # 3 samples vs 100 expected → drift 97% → dropped.
        assert payload is None
