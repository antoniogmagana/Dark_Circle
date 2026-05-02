"""
Comprehensive tests for SensorBuffer class.

Tests cover:
- Initialization and configuration
- Window boundary handling
- Multi-channel synchronization
- Holding pen logic for overflow data
- ADC normalization
- Timestamp handling and drift
- Protobuf packaging
- Edge cases (early data, late data, split data)
"""

import numpy as np
import pytest
from ingestor.buffer import SensorBuffer

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
    def test_init_sample_rates(self, sensor_id):
        """Test sample rates are correctly configured."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.rates["acoustic"] == 16000
        assert buffer.rates["seismic"] == 100
        assert buffer.rates["accel_x"] == 100
        assert buffer.rates["accel_y"] == 100
        assert buffer.rates["accel_z"] == 100

    @pytest.mark.unit
    def test_init_buffer_sizes(self, sensor_id):
        """Test buffer sizes match sample rates."""
        buffer = SensorBuffer(sensor_id)

        assert buffer.limits["acoustic"] == 16000
        assert buffer.limits["seismic"] == 100
        assert buffer.limits["accel_x"] == 100
        assert buffer.limits["accel_y"] == 100
        assert buffer.limits["accel_z"] == 100

        # Buffers should be pre-allocated
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


# ============================================================================
# Test Basic Loading
# ============================================================================


class TestBasicLoading:
    """Test basic data loading functionality."""

    @pytest.mark.unit
    def test_load_first_data(self, sensor_id, base_timestamp):
        """Test loading first data initializes start_time."""
        buffer = SensorBuffer(sensor_id)
        data = np.array([100, 200, 300])

        result = buffer.load_buffer("acoustic", base_timestamp, data)

        assert buffer.start_time == base_timestamp
        assert result is None
        assert "acoustic" in buffer.active_channels

    @pytest.mark.unit
    def test_load_early_data_rejected(self, sensor_id, base_timestamp):
        """Test data before window start is rejected."""
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        # Try to load data 0.1s before window start
        early_data = np.array([100, 200, 300])
        result = buffer.load_buffer("acoustic", base_timestamp - 0.1, early_data)

        assert result is None
        assert "acoustic" not in buffer.active_channels

    @pytest.mark.unit
    def test_load_channels_registered(self, sensor_id, base_timestamp):
        """Test channels are registered when loaded."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.array([1, 2, 3]))
        assert "acoustic" in buffer.active_channels

        buffer.load_buffer("seismic", base_timestamp + 0.1, np.array([4, 5]))
        assert "seismic" in buffer.active_channels

        assert len(buffer.active_channels) == 2

    @pytest.mark.unit
    def test_load_data_at_correct_position(self, sensor_id, base_timestamp):
        """Test data is loaded at correct buffer position based on timestamp."""
        buffer = SensorBuffer(sensor_id)

        # Load 100 samples at t=0 (should go to index 0)
        data1 = np.full(100, 42.0)
        buffer.load_buffer("acoustic", base_timestamp, data1)

        assert np.all(buffer.buffers["acoustic"][0:100] == 42.0)
        assert np.all(buffer.buffers["acoustic"][100:200] == 0.0)

        # Load 100 samples at t=0.1 (should go to index 1600)
        data2 = np.full(100, 99.0)
        buffer.load_buffer("acoustic", base_timestamp + 0.1, data2)

        assert np.all(buffer.buffers["acoustic"][1600:1700] == 99.0)


# ============================================================================
# Test Window Boundary Handling
# ============================================================================


class TestWindowBoundaries:
    """Test handling of data spanning window boundaries."""

    @pytest.mark.unit
    def test_acoustic_triggers_window_package(self, sensor_id, base_timestamp):
        """Test acoustic data crossing boundary triggers packaging."""
        buffer = SensorBuffer(sensor_id)

        # Fill most of first window
        data1 = np.full(15900, 100.0)
        result1 = buffer.load_buffer("acoustic", base_timestamp, data1)
        assert result1 is None

        # Add data that crosses boundary (150 samples total)
        # 100 samples fit in first window, 50 go to holding pen
        data2 = np.full(150, 200.0)
        result2 = buffer.load_buffer("acoustic", base_timestamp + 0.99375, data2)

        # Should return packaged window
        assert result2 is not None
        assert buffer.start_time == base_timestamp + 1.0  # Advanced

    @pytest.mark.unit
    def test_non_acoustic_crosses_to_holding_pen(self, sensor_id, base_timestamp):
        """Test non-acoustic data crossing boundary goes to holding pen."""
        buffer = SensorBuffer(sensor_id)

        # Initialize window
        buffer.load_buffer("acoustic", base_timestamp, np.array([1, 2, 3]))

        # Load seismic data that crosses boundary
        # At t=0.995, index=99 (only 1 sample fits in 100-sample buffer)
        data = np.full(10, 50.0)
        result = buffer.load_buffer("seismic", base_timestamp + 0.995, data)

        assert result is None  # Non-acoustic doesn't trigger packaging
        assert "seismic" in buffer.active_channels

        # 1 sample should be in buffer, 9 in holding pen
        assert buffer.buffers["seismic"][99] == 50.0
        assert len(buffer.holding_pen["seismic"]) == 9

    @pytest.mark.unit
    def test_holding_pen_transferred_on_reset(self, sensor_id, base_timestamp):
        """Test holding pen data is transferred to next window."""
        buffer = SensorBuffer(sensor_id)

        # Load acoustic to initialize
        buffer.load_buffer("acoustic", base_timestamp, np.full(15900, 1.0))

        # Load seismic that crosses boundary
        buffer.load_buffer("seismic", base_timestamp + 0.995, np.full(10, 99.0))

        # Trigger window package with acoustic
        acoustic_boundary = np.full(200, 2.0)
        buffer.load_buffer("acoustic", base_timestamp + 0.99, acoustic_boundary)

        # After reset, holding pen should be transferred
        assert len(buffer.holding_pen["seismic"]) == 0
        assert buffer.buffers["seismic"][0] == 99.0
        assert "seismic" in buffer.active_channels  # Re-registered

    @pytest.mark.unit
    def test_acoustic_split_across_boundary(self, sensor_id, base_timestamp):
        """Test acoustic data split across boundary."""
        buffer = SensorBuffer(sensor_id)

        # Fill to near capacity
        buffer.load_buffer("acoustic", base_timestamp, np.full(15950, 10.0))

        # Add 100 samples: 50 fit in window, 50 go to holding pen
        data = np.full(100, 20.0)
        payload = buffer.load_buffer("acoustic", base_timestamp + 0.996875, data)

        # Should package and return
        assert payload is not None

        # Holding pen should have 50 samples for next window
        assert len(buffer.holding_pen["acoustic"]) == 0  # Already transferred
        assert buffer.buffers["acoustic"][0] == 20.0  # First sample of chunk_2


# ============================================================================
# Test Multi-Channel Synchronization
# ============================================================================


class TestMultiChannel:
    """Test multi-channel data handling and synchronization."""

    @pytest.mark.unit
    def test_multiple_channels_same_window(self, sensor_id, base_timestamp):
        """Test loading multiple channels in same window."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.full(1000, 1.0))
        buffer.load_buffer("seismic", base_timestamp + 0.1, np.full(10, 2.0))
        buffer.load_buffer("accel_x", base_timestamp + 0.2, np.full(5, 3.0))

        assert len(buffer.active_channels) == 3
        assert buffer.buffers["acoustic"][0] == 1.0
        assert buffer.buffers["seismic"][10] == 2.0  # At t=0.1, index=10
        assert buffer.buffers["accel_x"][20] == 3.0  # At t=0.2, index=20

    @pytest.mark.unit
    def test_channels_with_different_rates(self, sensor_id, base_timestamp):
        """Test channels with different sample rates sync correctly."""
        buffer = SensorBuffer(sensor_id)

        # At t=0: acoustic index 0, seismic index 0
        buffer.load_buffer("acoustic", base_timestamp, np.array([100.0]))
        buffer.load_buffer("seismic", base_timestamp, np.array([200.0]))

        # At t=0.5: acoustic index 8000, seismic index 50
        buffer.load_buffer("acoustic", base_timestamp + 0.5, np.array([101.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.5, np.array([201.0]))

        assert buffer.buffers["acoustic"][0] == 100.0
        assert buffer.buffers["acoustic"][8000] == 101.0
        assert buffer.buffers["seismic"][0] == 200.0
        assert buffer.buffers["seismic"][50] == 201.0

    @pytest.mark.unit
    def test_accelerometer_three_axes(self, sensor_id, base_timestamp):
        """Test all three accelerometer axes can be loaded."""
        buffer = SensorBuffer(sensor_id)

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
# Test ADC Normalization
# ============================================================================


class TestPackageOutput:
    """Test the per-window output contract emitted by ``_package_window``.

    Two modes are exercised:
      * Default (``ADC_SCALE_NORMALIZE`` unset / 0) — raw ADC counts preserved,
        only per-window mean subtraction. This is the CRL training contract.
      * Legacy (``ADC_SCALE_NORMALIZE=1``) — divide by ``2^(bits-1)`` before
        mean subtraction. This is the pre-CRL ``WaveformClassificationCNN``
        contract; kept for back-compat behind the env flag.

    Tests drive ``_package_window`` directly by populating ``buffer.buffers``
    and ``buffer.active_channels`` so the assertion is on the package math,
    not on the streaming loader's window-boundary bookkeeping.
    """

    def _ramp(self, n, lo, hi):
        """Return n integer samples evenly spaced in [lo, hi]."""
        return np.linspace(lo, hi, n).astype(np.float64)

    @pytest.mark.unit
    def test_acoustic_default_preserves_raw_counts(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        """Default path: only per-window mean subtraction; raw scale preserved."""
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", False)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["acoustic"], -32768, 32767)
        buffer.buffers["acoustic"] = raw.copy()
        buffer.active_channels.add("acoustic")

        payload = buffer._package_window()
        out = np.array(payload.acoustic_data.data)

        # Mean subtraction only; std stays in raw-count range (~thousands).
        assert abs(out.mean()) < 1e-6
        np.testing.assert_allclose(out, raw - raw.mean(), atol=1e-3)
        assert out.std() > 1000  # raw-count scale, not [-1, 1]

    @pytest.mark.unit
    def test_acoustic_legacy_normalizes_to_unit_range(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        """ADC_SCALE_NORMALIZE=1: divide by 2^15, then mean-subtract."""
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", True)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["acoustic"], -32768, 32767)
        buffer.buffers["acoustic"] = raw.copy()
        buffer.active_channels.add("acoustic")

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
        buffer.active_channels.add("seismic")

        payload = buffer._package_window()
        out = np.array(payload.seismic_data.data)

        assert abs(out.mean()) < 1e-3
        np.testing.assert_allclose(out, raw - raw.mean(), atol=1e-2)
        assert out.std() > 1e5  # raw 24-bit scale

    @pytest.mark.unit
    def test_seismic_legacy_normalizes_to_unit_range(
        self, sensor_id, base_timestamp, skip_if_no_protos, monkeypatch
    ):
        monkeypatch.setattr("ingestor.buffer._ADC_SCALE_NORMALIZE", True)
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp

        raw = self._ramp(buffer.limits["seismic"], -8388608, 8388607)
        buffer.buffers["seismic"] = raw.copy()
        buffer.active_channels.add("seismic")

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
        for ax in ("accel_x", "accel_y", "accel_z"):
            buffer.active_channels.add(ax)

        payload = buffer._package_window()
        accel = np.array(payload.accel_data.data).reshape((3, n))
        # Per-axis constant input → mean-subtracted output is all zeros.
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
        for ax in ("accel_x", "accel_y", "accel_z"):
            buffer.active_channels.add(ax)

        payload = buffer._package_window()
        accel = np.array(payload.accel_data.data).reshape((3, n))
        scaled = ramp / (2**23)
        expected = scaled - scaled.mean()
        np.testing.assert_allclose(accel[0], expected, atol=1e-7)
        np.testing.assert_allclose(accel[1], expected, atol=1e-7)
        np.testing.assert_allclose(accel[2], expected, atol=1e-7)
        assert abs(accel.max()) < 1.01


# ============================================================================
# Test Protobuf Packaging
# ============================================================================


class TestProtobufPackaging:
    """Test protobuf message packaging."""

    @pytest.mark.unit
    def test_package_sensor_id(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test sensor ID is correctly set in package."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        assert payload is not None
        assert payload.sensor_id == sensor_id

    @pytest.mark.unit
    def test_package_timestamp(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test timestamp is correctly set in package."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        # Timestamp should match window start
        assert payload is not None
        assert payload.time_stamp.seconds == int(base_timestamp)
        assert payload.time_stamp.nanos == int((base_timestamp - int(base_timestamp)) * 1e9)

    @pytest.mark.unit
    def test_package_channels_list(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test channels list matches active channels."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        buffer.load_buffer("seismic", base_timestamp, np.zeros(100))
        buffer.load_buffer("accel_x", base_timestamp, np.zeros(100))

        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        assert payload is not None
        assert "acoustic" in payload.channels
        assert "seismic" in payload.channels
        assert "accel" in payload.channels  # Note: accel, not accel_x

    @pytest.mark.unit
    def test_package_tensor_shapes(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test tensor shapes in package."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        buffer.load_buffer("seismic", base_timestamp, np.zeros(100))
        buffer.load_buffer("accel_x", base_timestamp, np.zeros(100))
        buffer.load_buffer("accel_y", base_timestamp, np.zeros(100))
        buffer.load_buffer("accel_z", base_timestamp, np.zeros(100))

        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        # Acoustic: [16000]
        assert payload is not None
        assert payload.acoustic_data.shape == [16000]

        # Seismic: [100]
        assert payload.seismic_data.shape == [100]

        # Accel: [3, 100]
        assert payload.accel_data.shape == [3, 100]

    @pytest.mark.unit
    def test_package_only_active_channels(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test only active channels are included in package."""
        buffer = SensorBuffer(sensor_id)

        # Only load acoustic - crossing boundary returns payload
        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        assert payload is not None
        assert "acoustic" in payload.channels
        assert "seismic" not in payload.channels
        assert "accel" not in payload.channels


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_exact_window_boundary(self, sensor_id, base_timestamp):
        """Test data ending exactly at window boundary."""
        buffer = SensorBuffer(sensor_id)

        # Load data that ends exactly at boundary (15999 samples)
        # At 16kHz, this fills indices 0-15998, leaving one sample unfilled
        data = np.zeros(15999)
        result = buffer.load_buffer("acoustic", base_timestamp, data)

        # Should not trigger packaging (not crossing boundary)
        assert result is None

        # Loading one more sample should cross boundary and trigger packaging
        result2 = buffer.load_buffer("acoustic", base_timestamp + 0.9999375, np.array([1.0]))
        assert result2 is not None  # Should return payload

    @pytest.mark.unit
    def test_data_far_future_packages_active_window(self, sensor_id, base_timestamp):
        """A >1s gap with samples already in the current window packages it
        once, then restarts on the new timestamp without emitting phantom
        zero-windows for the silent seconds in between.

        Call signature matches the production caller in
        ``ingestor.dispatch.make_role_callback``:
        ``load_buffer(channel, timestamp, data)``.
        """
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(100))

        # 2-second gap. First call packages the active window (samples are
        # in it) and re-enters; the re-entry hits the stream-restart guard
        # because the buffers were just reset and the new timestamp is far
        # ahead, so start_time snaps to the new timestamp.
        result = buffer.load_buffer("acoustic", base_timestamp + 2.0, np.zeros(100))

        assert result is not None
        assert buffer.start_time == base_timestamp + 2.0

    @pytest.mark.unit
    def test_data_far_future_after_silence_does_not_phantom_publish(
        self,
        sensor_id,
        base_timestamp,
    ):
        """If the buffer is fresh (no samples loaded yet), a far-future
        timestamp must not emit a payload — there is nothing to publish."""
        buffer = SensorBuffer(sensor_id)
        buffer.start_time = base_timestamp  # simulate prior reset, no data

        result = buffer.load_buffer("acoustic", base_timestamp + 5.0, np.zeros(100))

        assert result is None
        assert buffer.start_time == base_timestamp + 5.0
        assert "acoustic" in buffer.active_channels

    @pytest.mark.unit
    def test_single_sample(self, sensor_id, base_timestamp):
        """Test loading single samples."""
        buffer = SensorBuffer(sensor_id)

        result = buffer.load_buffer("acoustic", base_timestamp, np.array([42.0]))

        assert result is None
        assert buffer.buffers["acoustic"][0] == 42.0

    @pytest.mark.unit
    def test_empty_array(self, sensor_id, base_timestamp):
        """Test loading empty array doesn't crash."""
        buffer = SensorBuffer(sensor_id)

        result = buffer.load_buffer("acoustic", base_timestamp, np.array([]))

        # Should handle gracefully
        assert result is None

    @pytest.mark.unit
    def test_reset_clears_active_channels(self, sensor_id, base_timestamp):
        """Test reset clears active channels."""
        buffer = SensorBuffer(sensor_id)

        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))
        buffer.load_buffer("seismic", base_timestamp, np.zeros(90))

        assert len(buffer.active_channels) == 2

        # Trigger reset
        buffer.load_buffer("acoustic", base_timestamp + 0.99, np.zeros(200))

        # Channels should be cleared (except those restored from holding pen)
        # In this case, no holding pen data, so should be empty or just acoustic
        assert len(buffer.active_channels) <= 1

    @pytest.mark.unit
    def test_multiple_windows_sequence(self, sensor_id, base_timestamp):
        """Test processing multiple windows in sequence."""
        buffer = SensorBuffer(sensor_id)

        payloads = []

        # Window 1 - fill and trigger packaging
        buffer.load_buffer("acoustic", base_timestamp, np.full(15900, 1.0))
        p1 = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.full(200, 1.0))
        payloads.append(p1)

        # Window 2 - fill and trigger packaging
        buffer.load_buffer("acoustic", base_timestamp + 1.0, np.full(15900, 2.0))
        p2 = buffer.load_buffer("acoustic", base_timestamp + 1.99375, np.full(200, 2.0))
        payloads.append(p2)

        # Window 3 - fill and trigger packaging
        buffer.load_buffer("acoustic", base_timestamp + 2.0, np.full(15900, 3.0))
        p3 = buffer.load_buffer("acoustic", base_timestamp + 2.99375, np.full(200, 3.0))
        payloads.append(p3)

        assert len(payloads) == 3
        assert all(p is not None for p in payloads)
        assert buffer.start_time == 3.0


# ============================================================================
# Test Complex Scenarios
# ============================================================================


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    @pytest.mark.unit
    def test_realistic_acoustic_chunks(self, sensor_id, base_timestamp):
        """Test realistic acoustic data arrival (e.g., 100-sample chunks)."""
        buffer = SensorBuffer(sensor_id)

        # Simulate 159 chunks of 100 samples each first
        for i in range(159):
            timestamp = base_timestamp + (i * 100.0 / 16000.0)  # Time for each chunk
            data = np.full(100, i)
            result = buffer.load_buffer("acoustic", timestamp, data)
            assert result is None, f"Chunk {i} should not trigger packaging"

        # 160th chunk (final 100 samples) should cross boundary and trigger packaging
        final_timestamp = base_timestamp + (159 * 100.0 / 16000.0)
        final_result = buffer.load_buffer("acoustic", final_timestamp, np.full(100, 159))
        assert final_result is not None, "Final chunk should trigger packaging"

    @pytest.mark.unit
    def test_interleaved_multi_channel(self, sensor_id, base_timestamp):
        """Test realistic interleaved multi-channel data."""
        buffer = SensorBuffer(sensor_id)

        # Simulate interleaved arrival
        for t in np.arange(0, 0.9, 0.01):  # 0 to 0.9s in 10ms steps
            ts = base_timestamp + t

            # Acoustic: 160 samples every 10ms
            buffer.load_buffer("acoustic", ts, np.random.randn(160))

            # Seismic: 1 sample every 10ms
            buffer.load_buffer("seismic", ts, np.array([t]))

            # Accel: 1 sample every 10ms per axis
            buffer.load_buffer("accel_x", ts, np.array([t]))
            buffer.load_buffer("accel_y", ts, np.array([t * 2]))
            buffer.load_buffer("accel_z", ts, np.array([t * 3]))

        # All channels should be active
        assert len(buffer.active_channels) == 5

    @pytest.mark.unit
    def test_sparse_seismic_arrival(self, sensor_id, base_timestamp, skip_if_no_protos):
        """Test sparse seismic data (low frequency) with dense acoustic."""
        buffer = SensorBuffer(sensor_id)

        # Fill acoustic continuously - use data that will cross boundary
        buffer.load_buffer("acoustic", base_timestamp, np.zeros(15900))

        # Add seismic at sparse intervals
        buffer.load_buffer("seismic", base_timestamp + 0.0, np.array([1.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.5, np.array([2.0]))
        buffer.load_buffer("seismic", base_timestamp + 0.9, np.array([3.0]))

        # Cross boundary to trigger packaging
        payload = buffer.load_buffer("acoustic", base_timestamp + 0.99375, np.zeros(200))

        # Seismic buffer should have 3 non-zero values at correct positions.
        # Default path preserves raw ADC counts (CRL contract); only mean
        # subtraction is applied. Per-window mean of [1,2,3] over 100 samples
        # is 0.06, so values land at {0.94, 1.94, 2.94}. Tolerance accounts
        # for the protobuf float32 round-trip on the way out.
        assert payload is not None
        assert payload.HasField("seismic_data")
        out = payload.seismic_data.data
        expected_mean = (1.0 + 2.0 + 3.0) / 100
        assert abs(out[0] - (1.0 - expected_mean)) < 1e-6
        assert abs(out[50] - (2.0 - expected_mean)) < 1e-6
        assert abs(out[90] - (3.0 - expected_mean)) < 1e-6
