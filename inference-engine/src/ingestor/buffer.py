import os

import numpy as np
import torch
import torchaudio.functional as F
from google.protobuf.timestamp_pb2 import Timestamp
from inference_protos import inference_pb2

# Roles the buffer accepts. Source of truth (dispatch.py imports these).
VALID_ROLES = frozenset({"acoustic", "seismic", "accel_x", "accel_y", "accel_z"})
REQUIRED_ROLES = frozenset({"acoustic", "seismic"})
ACCEL_ROLES = frozenset({"accel_x", "accel_y", "accel_z"})

# Tolerance for window-close rate validation: realized samples-per-window may
# legitimately drift this fraction from expected before we drop the window.
RATE_TOLERANCE = 0.01

_AUDIO_BITS = int(os.environ.get("AUDIO_BIT_DEPTH", "16"))
_SEISMIC_BITS = int(os.environ.get("SEISMIC_BIT_DEPTH", "24"))
_ACCEL_BITS = int(os.environ.get("ACCEL_BIT_DEPTH", "24"))

# CRL was trained on raw ADC counts with mean subtraction only (parquets store
# int amplitudes cast to float32; see crl-train/crl_vehicle/data/dataset.py and
# server-load/sample_parse.py). The legacy WaveformClassificationCNN path
# expected [-1, 1] floats. Default to preserving raw counts; set
# ADC_SCALE_NORMALIZE=1 to reinstate the legacy behaviour.
_ADC_SCALE_NORMALIZE = os.environ.get("ADC_SCALE_NORMALIZE", "0") == "1"


class SensorBuffer:
    def __init__(self, sensor_id, expected_rates=None):
        self.active_channels = set()
        self.sensor_id = sensor_id
        self.start_time = None
        self.window = 1.0

        # Per-channel native (incoming) sample rates. ``expected_rates`` is
        # the channel-map's authority on rates; fall back to the (legacy)
        # default rates only when no map was supplied.
        default_rates = {
            "acoustic": 16000,
            "seismic": 100,
            "accel_x": 100,
            "accel_y": 100,
            "accel_z": 100,
        }
        if expected_rates is None:
            self.rates = default_rates
        else:
            self.rates = {ch: expected_rates.get(ch, default_rates[ch]) for ch in default_rates}

        # Per-channel TARGET (post-resample) rates. The buffer resamples each
        # incoming channel to its target rate before packaging.
        # When NATIVE_RATES=1 we ship every channel at its native rate
        # (CRL expects audio=16k, seismic=100). Otherwise everything goes
        # through the legacy "upsample to TARGET_RATE" path that the older
        # WaveformClassificationCNN model expected.
        if os.environ.get("NATIVE_RATES", "0") == "1":
            self.target_rates = dict(self.rates)
        else:
            tr = int(os.environ.get("TARGET_RATE", "16000"))
            self.target_rates = {ch: tr for ch in self.rates}

        self.limits = {ch: int(self.window * self.target_rates[ch]) for ch in self.rates}
        self.buffers = {ch: np.zeros(self.limits[ch]) for ch in self.rates}

        # received_samples counts how many samples actually arrived for each
        # channel during the current window, including any carried over from
        # the holding pen during _reset_buffers. _package_window compares
        # this against the expected total (``window * target_rate``) and
        # drops the window if the realized count drifts outside tolerance.
        self.received_samples = dict.fromkeys(default_rates, 0)

        self.holding_pen = {
            "acoustic": [],
            "seismic": [],
            "accel_x": [],
            "accel_y": [],
            "accel_z": [],
        }
        self.adc_scale = {
            "acoustic": 2 ** (_AUDIO_BITS - 1),
            "seismic": 2 ** (_SEISMIC_BITS - 1),
            "accel": 2 ** (_ACCEL_BITS - 1),
        }

    def _resample(self, signal, channel):
        # Incoming ROS2 int32[] arrives as array.array; torchaudio needs a Tensor.
        tensor = torch.as_tensor(signal, dtype=torch.float32)
        out = F.resample(tensor, self.rates[channel], self.target_rates[channel])
        return out.numpy()

    def _package_window(self):
        # Window-close rate validation. For each active channel, compare
        # realized samples against window*target_rate. A drift outside
        # ±RATE_TOLERANCE drops the window: better to miss one second than
        # ship rate-corrupted data into the model.
        for ch in self.active_channels:
            expected = self.limits[ch]
            got = self.received_samples[ch]
            if expected == 0:
                continue
            drift = abs(got - expected) / expected
            if drift > RATE_TOLERANCE:
                print(
                    f"[buffer:{ch}] window dropped: rate mismatch "
                    f"(got {got} samples, expected {expected})",
                    flush=True,
                )
                return None

        ts = Timestamp()
        ts.seconds = int(self.start_time)
        ts.nanos = int((self.start_time - int(self.start_time)) * 1e9)

        payload = inference_pb2.SensorData(sensor_id=self.sensor_id, time_stamp=ts)

        if "acoustic" in self.active_channels:
            arr = self.buffers["acoustic"]
            if _ADC_SCALE_NORMALIZE:
                arr = arr / self.adc_scale["acoustic"]
            arr = arr - arr.mean()

            payload.channels.append("acoustic")
            payload.acoustic_data.CopyFrom(
                inference_pb2.Tensor(shape=list(arr.shape), data=arr.flatten().tolist())
            )

        if "seismic" in self.active_channels:
            arr = self.buffers["seismic"]
            if _ADC_SCALE_NORMALIZE:
                arr = arr / self.adc_scale["seismic"]
            arr = arr - arr.mean()

            payload.channels.append("seismic")
            payload.seismic_data.CopyFrom(
                inference_pb2.Tensor(shape=list(arr.shape), data=arr.flatten().tolist())
            )

        if any(axis in self.active_channels for axis in ["accel_x", "accel_y", "accel_z"]):
            payload.channels.append("accel")
            accel_matrix = np.vstack(
                (
                    self.buffers["accel_x"],
                    self.buffers["accel_y"],
                    self.buffers["accel_z"],
                )
            )
            if _ADC_SCALE_NORMALIZE:
                accel_matrix = accel_matrix / self.adc_scale["accel"]
            accel_matrix = accel_matrix - accel_matrix.mean(axis=1, keepdims=True)

            payload.accel_data.CopyFrom(
                inference_pb2.Tensor(
                    shape=list(accel_matrix.shape),
                    data=accel_matrix.flatten().tolist(),
                )
            )
        return payload

    def _reset_buffers(self):
        self.active_channels.clear()
        self.received_samples = dict.fromkeys(self.buffers, 0)

        for ch in self.buffers:
            self.buffers[ch].fill(0)
            limit = self.limits[ch]
            pen = self.holding_pen[ch]
            pen_len = len(pen)

            # Capping the holding pen by the buffer size keeps a runaway
            # publisher (incoming rate above configured rate) from growing
            # the pen without bound.
            if pen_len > 0:
                if pen_len > limit:
                    pen = pen[-limit:]
                    pen_len = limit
                self.buffers[ch][0:pen_len] = pen
                self.holding_pen[ch].clear()
                self.active_channels.add(ch)
                self.received_samples[ch] += pen_len

        self.start_time += self.window
        return None

    def maybe_close_window(self, timestamp):
        """Decide whether ``timestamp`` ends the current window.

        Called once per incoming bundled message, before any channel data
        is written. Returns a packaged ``SensorData`` payload when a window
        closes (or ``None`` if the rate-mismatch guard rejected it), and
        ``None`` when the new message still belongs to the current window.

        Replaces the previous "acoustic channel triggers close" rule, which
        only worked when channels arrived on independent topics.
        """
        if self.start_time is None:
            self.start_time = timestamp
            return None

        time_diff = timestamp - self.start_time
        if time_diff < self.window:
            return None

        # Stream-restart guard: if no channel has loaded any sample into
        # the current window AND no holding-pen carry is queued, the
        # source has been silent. Snap start_time to the new timestamp
        # rather than emit phantom zero-windows for every silent second.
        if not self.active_channels and not any(self.holding_pen.values()):
            self.start_time = timestamp
            return None

        ready_payload = self._package_window()
        # _reset_buffers advances start_time by exactly self.window, not
        # to the new message's timestamp. This preserves holding-pen
        # carry semantics (chunk_2 from the previous window lands at
        # offset 0 of the new window). After a long gap, the next few
        # windows may drop on the rate-mismatch guard while start_time
        # catches up — that is intentional. Do NOT snap start_time to
        # ``timestamp`` here; the silent-stream guard above handles the
        # truly-idle case.
        self._reset_buffers()
        return ready_payload

    def load_buffer(self, channel, timestamp, data):
        """Write one channel's slice of a bundled message into the current window.

        Window-close logic moved to ``maybe_close_window``; the dispatch
        callback must call that first per message. ``load_buffer`` here only
        handles the data write and the within-message straddle case
        (chunk_1 into the current window, chunk_2 into the holding pen).
        """
        rate = self.target_rates[channel]
        limit = self.limits[channel]
        buffer = self.buffers[channel]

        if self.start_time is None:
            # maybe_close_window should have set this, but be defensive.
            self.start_time = timestamp

        time_diff = timestamp - self.start_time
        if time_diff < 0:
            return None

        self.active_channels.add(channel)
        if self.rates[channel] != self.target_rates[channel]:
            data = self._resample(data, channel)
        start_index = int(time_diff * rate)
        data_length = len(data)
        end_index = start_index + data_length

        if end_index < limit:
            buffer[start_index:end_index] = data
            self.received_samples[channel] += data_length
            return None

        # Straddle: chunk_1 lands in this window, chunk_2 carries to the next
        # via the holding pen. maybe_close_window will fire on the next
        # message and flush the pen during _reset_buffers.
        space_left = limit - start_index
        chunk_1 = data[0:space_left]
        chunk_2 = data[space_left:]
        buffer[start_index:limit] = chunk_1
        self.received_samples[channel] += space_left
        self.holding_pen[channel].extend(chunk_2)
        return None
