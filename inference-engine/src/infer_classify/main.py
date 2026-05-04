"""
Classification node — CRL TorchScript edition.

Loads (encoder + type head) for either fused or per_sensor mode from
``MODEL_DIR``. On a positive ``DetectionResult`` the pod re-encodes the
inbound raw waveform — the detect pod no longer ships a latent on the
wire — and applies the type head to produce ``vehicle_class`` and
``classification_confidence``.

For per-sensor mode we average the two heads' softmax probabilities
before argmax — gives a smooth fused decision without re-encoding.
"""

import asyncio
import json
import os
from pathlib import Path

import nats
import torch
import torch.nn.functional as F
from inference_protos import inference_pb2

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(_TORCH_THREADS)


def _load_artifacts(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    mode = meta["mode"]

    device = torch.device("cpu")
    encoders: dict[str, torch.jit.ScriptModule] = {}
    heads: dict[str, torch.jit.ScriptModule] = {}

    if mode == "fused":
        encoders["fused"] = torch.jit.load(
            str(model_dir / "encoder_fused.ts"), map_location=device
        ).eval()
        heads["fused"] = torch.jit.load(
            str(model_dir / "type_head_fused.ts"), map_location=device
        ).eval()
    elif mode == "per_sensor":
        for sensor in meta["sensors"]:
            encoders[sensor] = torch.jit.load(
                str(model_dir / f"encoder_{sensor}.ts"), map_location=device
            ).eval()
            heads[sensor] = torch.jit.load(
                str(model_dir / f"type_head_{sensor}.ts"), map_location=device
            ).eval()
    else:
        raise ValueError(f"unknown mode in meta.json: {mode!r}")

    print(
        f"[infer_classify] loaded mode={mode} sensors={meta['sensors']} "
        f"classes={meta['class_names']} z_dim={meta['z_dim']}",
        flush=True,
    )
    return encoders, heads, meta, device


def _to_tensor(values, expected_len: int) -> torch.Tensor:
    """Pack a flat list/array into shape (1, 1, expected_len) float32."""
    t = torch.as_tensor(list(values), dtype=torch.float32)
    if t.numel() != expected_len:
        raise ValueError(f"expected {expected_len} samples, got {t.numel()}")
    return t.view(1, 1, expected_len)


class InferClassifyNode:
    def __init__(self, nc, encoders, heads, meta, device):
        self.nc = nc
        self.js = nc.jetstream()
        self.encoders = encoders
        self.heads = heads
        self.meta = meta
        self.mode = meta["mode"]
        self.class_names = meta["class_names"]
        self.device = device

    def _infer_fused(self, x_audio: torch.Tensor, x_seismic: torch.Tensor):
        with torch.inference_mode():
            z, _pres_logit = self.encoders["fused"](x_audio, x_seismic)
            logits = self.heads["fused"](z)
        return F.softmax(logits, dim=1)[0]

    def _infer_per_sensor(self, x_audio: torch.Tensor, x_seismic: torch.Tensor):
        with torch.inference_mode():
            z_audio, _pres_audio = self.encoders["audio"](x_audio)
            z_seismic, _pres_seismic = self.encoders["seismic"](x_seismic)
            logits_audio = self.heads["audio"](z_audio)
            logits_seismic = self.heads["seismic"](z_seismic)
        probs_audio = F.softmax(logits_audio, dim=1)[0]
        probs_seismic = F.softmax(logits_seismic, dim=1)[0]
        return (probs_audio + probs_seismic) * 0.5

    async def on_detection_result(self, msg):
        detection = inference_pb2.DetectionResult()
        detection.ParseFromString(msg.data)

        if not detection.vehicle_detected:
            return

        sd = detection.sensor_data

        if not sd.HasField("acoustic_data") or not sd.HasField("seismic_data"):
            print(
                f"[infer_classify] {sd.sensor_id}: missing audio or seismic, skipping",
                flush=True,
            )
            return

        try:
            x_audio = _to_tensor(sd.acoustic_data.data, self.meta["audio_window_size"])
            x_seismic = _to_tensor(sd.seismic_data.data, self.meta["seismic_window_size"])
        except ValueError as exc:
            print(f"[infer_classify] shape mismatch: {exc}", flush=True)
            return

        loop = asyncio.get_event_loop()

        if self.mode == "fused":
            probs = await loop.run_in_executor(
                None, self._infer_fused, x_audio, x_seismic
            )
        else:
            probs = await loop.run_in_executor(
                None, self._infer_per_sensor, x_audio, x_seismic
            )

        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())
        vehicle_class = self.class_names[class_idx]

        payload = inference_pb2.EgressPayload()
        payload.sensor_id = sd.sensor_id
        payload.time_stamp.CopyFrom(sd.time_stamp)
        payload.vehicle_detected = True
        payload.detection_confidence = detection.confidence
        payload.vehicle_class = vehicle_class
        payload.classification_confidence = confidence

        await self.js.publish("classification.result", payload.SerializeToString())


async def main_async():
    if "NATS_URL" not in os.environ:
        raise OSError("Required environment variable 'NATS_URL' is not set")

    encoders, heads, meta, device = _load_artifacts(MODEL_DIR)

    nc = await nats.connect(os.environ["NATS_URL"])
    node = InferClassifyNode(nc, encoders, heads, meta, device)
    js = nc.jetstream()
    info = await js.consumer_info("DETECTION_RESULT", "infer-classify")
    await js.subscribe_bind(
        stream="DETECTION_RESULT",
        consumer="infer-classify",
        config=info.config,
        cb=node.on_detection_result,
        manual_ack=False,
    )
    print("[infer_classify] bound to DETECTION_RESULT/infer-classify (JetStream)", flush=True)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main_async())
