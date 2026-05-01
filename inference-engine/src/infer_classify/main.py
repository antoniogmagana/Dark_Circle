"""
Classification node — CRL TorchScript edition.

Loads either ``type_head_fused.ts`` (fused mode) or both
``type_head_audio.ts`` + ``type_head_seismic.ts`` (per-sensor mode) from
``MODEL_DIR``. Reads the latent ``z`` from the inbound ``DetectionResult``
(no encoder runs here) and produces an ``EgressPayload`` with the predicted
``vehicle_class`` and confidence.

For per-sensor mode we average the two heads' softmax probabilities before
argmax — gives a smooth fused decision without re-encoding.
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


def _load_artifacts(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    mode = meta["mode"]

    device = torch.device("cpu")
    heads: dict[str, torch.jit.ScriptModule] = {}

    if mode == "fused":
        heads["fused"] = torch.jit.load(
            str(model_dir / "type_head_fused.ts"), map_location=device
        ).eval()
    elif mode == "per_sensor":
        for sensor in meta["sensors"]:
            heads[sensor] = torch.jit.load(
                str(model_dir / f"type_head_{sensor}.ts"), map_location=device
            ).eval()
    else:
        raise ValueError(f"unknown mode in meta.json: {mode!r}")

    print(
        f"[infer_classify] loaded mode={mode} sensors={meta['sensors']} "
        f"classes={meta['class_names']}",
        flush=True,
    )
    return heads, meta, device


def _z_to_tensor(values, z_dim: int) -> torch.Tensor:
    t = torch.as_tensor(list(values), dtype=torch.float32)
    if t.numel() != z_dim:
        raise ValueError(f"expected z of length {z_dim}, got {t.numel()}")
    return t.view(1, z_dim)


class InferClassifyNode:
    def __init__(self, nc, heads, meta, device):
        self.nc = nc
        self.js = nc.jetstream()
        self.heads = heads
        self.meta = meta
        self.mode = meta["mode"]
        self.class_names = meta["class_names"]
        self.z_dim = int(meta["z_dim"])
        self.device = device

    def _infer_fused(self, z: torch.Tensor):
        with torch.inference_mode():
            logits = self.heads["fused"](z)
        return F.softmax(logits, dim=1)[0]

    def _infer_per_sensor(self, z_audio: torch.Tensor, z_seismic: torch.Tensor):
        with torch.inference_mode():
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

        loop = asyncio.get_event_loop()

        try:
            if self.mode == "fused":
                if not detection.z_fused:
                    print("[infer_classify] missing z_fused; skipping", flush=True)
                    return
                z = _z_to_tensor(detection.z_fused, self.z_dim)
                probs = await loop.run_in_executor(None, self._infer_fused, z)
            else:
                if not detection.z_audio or not detection.z_seismic:
                    print(
                        "[infer_classify] missing z_audio or z_seismic; skipping",
                        flush=True,
                    )
                    return
                z_audio = _z_to_tensor(detection.z_audio, self.z_dim)
                z_seismic = _z_to_tensor(detection.z_seismic, self.z_dim)
                probs = await loop.run_in_executor(None, self._infer_per_sensor, z_audio, z_seismic)
        except ValueError as exc:
            print(f"[infer_classify] z shape error: {exc}", flush=True)
            return

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

    heads, meta, device = _load_artifacts(MODEL_DIR)

    nc = await nats.connect(os.environ["NATS_URL"])
    node = InferClassifyNode(nc, heads, meta, device)
    js = nc.jetstream()
    await js.subscribe(
        "detection.result",
        queue="infer-classify",
        durable="infer-classify",
        cb=node.on_detection_result,
        manual_ack=False,
    )
    print("[infer_classify] subscribed to detection.result (JetStream)", flush=True)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main_async())
