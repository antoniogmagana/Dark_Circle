"""
Detection node — CRL TorchScript edition.

Loads either an ``encoder_fused.ts`` (fused mode) or a pair of
``encoder_audio.ts`` + ``encoder_seismic.ts`` files (per-sensor mode) from
``MODEL_DIR``. The mode + thresholds + window sizes come from
``meta.json`` written alongside the TorchScript artifacts by
``crl-train/export_for_inference.py``.

For each ``sensor.data`` window we publish a ``DetectionResult`` carrying
the original ``sensor_data`` plus ``vehicle_detected`` and ``confidence``.
The latent stays inside this pod — the classification node has its own
encoder bundle and re-encodes the raw waveform from ``sensor_data``.
"""

import asyncio
import json
import os
from pathlib import Path

import nats
import torch
from inference_protos import inference_pb2

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))

# Constrain PyTorch's intra-op thread pool to the container's CPU budget.
# Without this, PyTorch defaults to the host's physical core count (e.g.
# 120 on a beefy server), but the cgroup CPU limit (typically 2) throttles
# the process. Threads then thrash competing for the limited CPU slices,
# turning a 50ms inference into a 2-second one. Drop the limit here so
# PyTorch only spawns as many threads as it can actually run in parallel.
_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(_TORCH_THREADS)


def _load_artifacts(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    mode = meta["mode"]

    device = torch.device("cpu")
    encoders: dict[str, torch.jit.ScriptModule] = {}

    if mode == "fused":
        encoders["fused"] = torch.jit.load(
            str(model_dir / "encoder_fused.ts"), map_location=device
        ).eval()
    elif mode == "per_sensor":
        for sensor in meta["sensors"]:
            encoders[sensor] = torch.jit.load(
                str(model_dir / f"encoder_{sensor}.ts"), map_location=device
            ).eval()
    else:
        raise ValueError(f"unknown mode in meta.json: {mode!r}")

    print(
        f"[infer_detect] loaded mode={mode} sensors={meta['sensors']} "
        f"z_dim={meta['z_dim']} threshold={meta['presence_threshold']}",
        flush=True,
    )
    return encoders, meta, device


def _pres_threshold(meta: dict, sensor: str | None = None) -> float:
    """Resolve the sigmoid threshold for a given sensor (per-sensor mode)
    or the single fused threshold (fused mode)."""
    thr = meta["presence_threshold"]
    if isinstance(thr, dict):
        if sensor is None:
            raise ValueError("per-sensor threshold dict but no sensor name")
        return float(thr[sensor])
    return float(thr)


def _to_tensor(values, expected_len: int) -> torch.Tensor:
    """Pack a flat list/array into shape (1, 1, expected_len) float32."""
    t = torch.as_tensor(list(values), dtype=torch.float32)
    if t.numel() != expected_len:
        raise ValueError(f"expected {expected_len} samples, got {t.numel()}")
    return t.view(1, 1, expected_len)


class InferDetectNode:
    def __init__(self, nc, encoders, meta, device):
        self.nc = nc
        self.js = nc.jetstream()
        self.encoders = encoders
        self.meta = meta
        self.mode = meta["mode"]
        self.device = device

    def _infer_fused(self, x_audio: torch.Tensor, x_seismic: torch.Tensor):
        with torch.inference_mode():
            z, pres_logit = self.encoders["fused"](x_audio, x_seismic)
        prob = torch.sigmoid(pres_logit).squeeze().item()
        threshold = _pres_threshold(self.meta)
        detected = prob >= threshold
        return detected, prob, z[0]

    def _infer_per_sensor(self, x_audio: torch.Tensor, x_seismic: torch.Tensor):
        with torch.inference_mode():
            z_audio, pres_audio = self.encoders["audio"](x_audio)
            z_seismic, pres_seismic = self.encoders["seismic"](x_seismic)
        prob_audio = torch.sigmoid(pres_audio).squeeze().item()
        prob_seismic = torch.sigmoid(pres_seismic).squeeze().item()
        # OR fusion: detected if EITHER sensor crosses its own threshold.
        det_audio = prob_audio >= _pres_threshold(self.meta, "audio")
        det_seismic = prob_seismic >= _pres_threshold(self.meta, "seismic")
        detected = det_audio or det_seismic
        # Confidence: max of the two so the downstream sees the strongest signal.
        prob = max(prob_audio, prob_seismic)
        return detected, prob, z_audio[0], z_seismic[0]

    async def on_sensor_data(self, msg):
        sd = inference_pb2.SensorData()
        sd.ParseFromString(msg.data)

        if not sd.HasField("acoustic_data") or not sd.HasField("seismic_data"):
            print(
                f"[infer_detect] {sd.sensor_id}: missing audio or seismic, skipping",
                flush=True,
            )
            return

        try:
            x_audio = _to_tensor(sd.acoustic_data.data, self.meta["audio_window_size"])
            x_seismic = _to_tensor(sd.seismic_data.data, self.meta["seismic_window_size"])
        except ValueError as exc:
            print(f"[infer_detect] shape mismatch: {exc}", flush=True)
            return

        result = inference_pb2.DetectionResult()
        result.sensor_data.CopyFrom(sd)

        loop = asyncio.get_event_loop()

        if self.mode == "fused":
            detected, prob, _z = await loop.run_in_executor(
                None, self._infer_fused, x_audio, x_seismic
            )
        else:
            detected, prob, _z_audio, _z_seismic = await loop.run_in_executor(
                None, self._infer_per_sensor, x_audio, x_seismic
            )

        result.vehicle_detected = bool(detected)
        result.confidence = float(prob)

        await self.js.publish("detection.result", result.SerializeToString())


async def main_async():
    if "NATS_URL" not in os.environ:
        raise OSError("Required environment variable 'NATS_URL' is not set")

    encoders, meta, device = _load_artifacts(MODEL_DIR)

    nc = await nats.connect(os.environ["NATS_URL"])
    js = nc.jetstream()
    # Bind to the pre-created ``infer-detect`` durable consumer
    # (jetstream-init Job sets it up with deliver_group="infer-detect").
    # Binding rather than auto-creating sidesteps nats-py's client-side
    # ``queue != durable`` validation and KEDA scale-out config drift.
    node = InferDetectNode(nc, encoders, meta, device)
    info = await js.consumer_info("SENSOR_DATA", "infer-detect")
    await js.subscribe_bind(
        stream="SENSOR_DATA",
        consumer="infer-detect",
        config=info.config,
        cb=node.on_sensor_data,
        manual_ack=False,
    )
    print("[infer_detect] bound to SENSOR_DATA/infer-detect (JetStream)", flush=True)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main_async())
