import asyncio
import json
import os
from types import SimpleNamespace

import nats
import torch
import torch.nn.functional as F

from inference_protos import inference_pb2
from models import build_model


MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")


def _load_model(model_dir: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with open(os.path.join(model_dir, "hyperparameters.json")) as f:
        config_dict = json.load(f)
    config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}
    config = SimpleNamespace(**config_dict)

    meta = torch.load(
        os.path.join(model_dir, "meta.pt"), map_location=device, weights_only=False
    )
    config.USE_MEL = meta.get("use_mel", False)

    model = build_model(config.IN_CHANNELS, config.NUM_CLASSES, config).to(device)

    state_dict = torch.load(
        os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=True
    )
    # Strip torch.compile prefix if present
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"[infer_detect] Loaded {config.MODEL_NAME} "
        f"({config.IN_CHANNELS}ch, SEQ_LEN={config.SEQ_LEN}) on {device}",
        flush=True,
    )
    return model, config, device


class InferDetectNode:
    def __init__(self, nc, model, config, device):
        self.nc = nc
        self.model = model
        self.config = config
        self.device = device

    def _infer(self, x: torch.Tensor) -> tuple[bool, float]:
        """Blocking inference — runs in executor to avoid blocking the event loop."""
        with torch.inference_mode():
            logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        vehicle_detected = logits.argmax(dim=1).item() == 1
        confidence = probs[0, 1].item()
        return vehicle_detected, confidence

    async def on_sensor_data(self, msg):
        sd = inference_pb2.SensorData()
        sd.ParseFromString(msg.data)

        if not sd.HasField("acoustic_data") or not sd.HasField("seismic_data"):
            print(
                f"[WARN] {sd.sensor_id}: missing audio or seismic channel, skipping",
                flush=True,
            )
            return

        audio = torch.tensor(
            list(sd.acoustic_data.data), dtype=torch.float32
        ).reshape(1, 1, self.config.SEQ_LEN)
        seismic = torch.tensor(
            list(sd.seismic_data.data), dtype=torch.float32
        ).reshape(1, 1, self.config.SEQ_LEN)

        # Buffer already applied ADC normalization and mean subtraction.
        # Only clamp residual spikes before inference.
        x = torch.cat([audio, seismic], dim=1).to(self.device)
        x = torch.clamp(x, -10.0, 10.0)

        loop = asyncio.get_event_loop()
        vehicle_detected, confidence = await loop.run_in_executor(None, self._infer, x)

        result = inference_pb2.DetectionResult()
        result.sensor_data.CopyFrom(sd)
        result.vehicle_detected = vehicle_detected
        result.confidence = confidence

        await self.nc.publish("detection.result", result.SerializeToString())


async def main_async():
    for var in ["NATS_URL"]:
        if var not in os.environ:
            raise EnvironmentError(f"Required environment variable '{var}' is not set")

    model, config, device = _load_model(MODEL_DIR)

    nc = await nats.connect(os.environ["NATS_URL"])
    node = InferDetectNode(nc, model, config, device)

    await nc.subscribe("sensor.data", cb=node.on_sensor_data)
    print("[infer_detect] Subscribed to sensor.data", flush=True)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main_async())
