"""
Detection node — CRL TorchScript edition.

Loads either an ``encoder_fused.ts`` (fused mode) or a pair of
``encoder_audio.ts`` + ``encoder_seismic.ts`` files (per-sensor mode) from
``MODEL_DIR``. The mode + thresholds + window sizes come from
``meta.json`` written alongside the TorchScript artifacts by
``crl-train/export_for_inference.py``.

For each ``sensor.data`` window we publish a ``DetectionResult`` whose
``z_fused`` (or ``z_audio`` + ``z_seismic``) field carries the latent so
the classification node can apply the type head without re-encoding.
"""

import asyncio
import json
import os
from pathlib import Path

import nats
import torch
from inference_protos import inference_pb2

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))


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
            detected, prob, z = await loop.run_in_executor(
                None, self._infer_fused, x_audio, x_seismic
            )
            result.z_fused.extend(z.tolist())
        else:
            detected, prob, z_audio, z_seismic = await loop.run_in_executor(
                None, self._infer_per_sensor, x_audio, x_seismic
            )
            result.z_audio.extend(z_audio.tolist())
            result.z_seismic.extend(z_seismic.tolist())

        result.vehicle_detected = bool(detected)
        result.confidence = float(prob)

        await self.js.publish("detection.result", result.SerializeToString())


async def main_async():
    if "NATS_URL" not in os.environ:
        raise OSError("Required environment variable 'NATS_URL' is not set")

    encoders, meta, device = _load_artifacts(MODEL_DIR)

    nc = await nats.connect(os.environ["NATS_URL"])
    js = nc.jetstream()
    # Queue group + durable consumer means multiple replicas (under KEDA)
    # split work, and a restarted pod resumes from its last delivered seq
    # instead of replaying the whole stream.
    node = InferDetectNode(nc, encoders, meta, device)
    await js.subscribe(
        "sensor.data",
        queue="infer-detect",
        durable="infer-detect",
        cb=node.on_sensor_data,
        manual_ack=False,
    )
    print("[infer_detect] subscribed to sensor.data (JetStream)", flush=True)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main_async())
