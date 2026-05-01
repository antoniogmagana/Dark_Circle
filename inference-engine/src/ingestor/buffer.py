import os

import numpy as np
import torch
import torchaudio.functional as F
from google.protobuf.timestamp_pb2 import Timestamp

from inference_protos import inference_pb2

_AUDIO_BITS = int(os.environ.get("AUDIO_BIT_DEPTH", "16"))
_SEISMIC_BITS = int(os.environ.get("SEISMIC_BIT_DEPTH", "24"))
_ACCEL_BITS = int(os.environ.get("ACCEL_BIT_DEPTH", "24"))


class SensorBuffer:
    def __init__(self, sensor_id):
        self.active_channels = set()
        self.sensor_id = sensor_id
        self.start_time = None
        self.window = 1.0

        # Per-channel native (incoming) sample rates.
        self.rates = {
            "acoustic": int(os.environ.get("AUDIO_SAMPLE_RATE", "16000")),
            "seismic": int(os.environ.get("SEISMIC_SAMPLE_RATE", "100")),
            "accel_x": int(os.environ.get("ACCEL_SAMPLE_RATE", "100")),
            "accel_y": int(os.environ.get("ACCEL_SAMPLE_RATE", "100")),
            "accel_z": int(os.environ.get("ACCEL_SAMPLE_RATE", "100")),
        }

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

        self.limits = {
            ch: int(self.window * self.target_rates[ch]) for ch in self.rates
        }
        self.buffers = {ch: np.zeros(self.limits[ch]) for ch in self.rates}

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
        ts = Timestamp()
        ts.seconds = int(self.start_time)
        ts.nanos = int((self.start_time - int(self.start_time)) * 1e9)

        payload = inference_pb2.SensorData(sensor_id=self.sensor_id, time_stamp=ts)

        if "acoustic" in self.active_channels:
            arr = self.buffers["acoustic"] / self.adc_scale["acoustic"]
            arr = arr - arr.mean()

            payload.channels.append("acoustic")
            payload.acoustic_data.CopyFrom(
                inference_pb2.Tensor(shape=list(arr.shape), data=arr.flatten().tolist())
            )

        if "seismic" in self.active_channels:
            arr = self.buffers["seismic"] / self.adc_scale["seismic"]
            arr = arr - arr.mean()

            payload.channels.append("seismic")
            payload.seismic_data.CopyFrom(
                inference_pb2.Tensor(shape=list(arr.shape), data=arr.flatten().tolist())
            )

        if any(
            axis in self.active_channels for axis in ["accel_x", "accel_y", "accel_z"]
        ):
            payload.channels.append("accel")
            accel_matrix = (
                np.vstack(
                    (
                        self.buffers["accel_x"],
                        self.buffers["accel_y"],
                        self.buffers["accel_z"],
                    )
                )
                / self.adc_scale["accel"]
            )
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

        self.start_time += self.window
        return None

    def load_buffer(self, channel, timestamp, data):
        rate = self.target_rates[channel]
        limit = self.limits[channel]
        buffer = self.buffers[channel]

        if self.start_time is None:
            self.start_time = timestamp

        time_diff = timestamp - self.start_time

        if time_diff < 0:
            return None

        if time_diff >= self.window:
            # Stream-restart guard: if no channel has loaded any sample into
            # the current window AND no holding-pen carry is queued, the
            # source has been silent. Snap start_time to the new timestamp
            # rather than emit phantom zero-windows for every silent second.
            if not self.active_channels and not any(self.holding_pen.values()):
                self.start_time = timestamp
                time_diff = 0.0
            elif channel == "acoustic":
                ready_payload = self._package_window()
                self._reset_buffers()
                self.load_buffer(channel, timestamp, data)
                return ready_payload
            else:
                self.holding_pen[channel].extend(data)
                return None

        self.active_channels.add(channel)
        if self.rates[channel] != self.target_rates[channel]:
            data = self._resample(data, channel)
        start_index = int(time_diff * rate)
        data_length = len(data)
        end_index = start_index + data_length

        if end_index < limit:
            buffer[start_index:end_index] = data
            return None

        space_left = limit - start_index
        chunk_1 = data[0:space_left]
        chunk_2 = data[space_left:]
        buffer[start_index:limit] = chunk_1

        if channel == "acoustic":
            ready_payload = self._package_window()
            self.holding_pen[channel].extend(chunk_2)
            self._reset_buffers()
            return ready_payload
        self.holding_pen[channel].extend(chunk_2)
        return None
