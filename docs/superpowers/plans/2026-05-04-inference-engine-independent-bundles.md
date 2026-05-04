# Independent Detect / Classify CRL Bundles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `inference-engine/` so `infer-detect` and `infer-classify` carry independently selected (backbone + head) bundles. Classify re-encodes from raw waveform; the latent fields disappear from `DetectionResult`.

**Architecture:** Two parallel bundle catalogs (`detect-bundles/`, `classify-bundles/`) replace the single `crl-bundles/`. Each bundle is self-contained — it has its own copy of the encoder it pairs with. The build script stages each pod's bundle independently into `build/detect-export/` and `build/classify-export/`. The crl-train exporter grows a `--bundle-kind {detect,classify}` flag that determines which artifacts and which `meta.json` fields the bundle gets, and a `--promote-default` flag that re-evaluates the catalog and repoints the `*-default` symlink against the selection rules.

**Tech Stack:** Python 3.12, PyTorch (TorchScript), protobuf 3, NATS / JetStream, poetry, Docker, Helm 3, kind, pytest.

**Spec:** `docs/superpowers/specs/2026-05-04-inference-engine-independent-bundles-design.md`

---

## File Structure

### Files created

- `inference-engine/detect-bundles/README.md` — catalog table for the detect-bundle catalog and producing-a-bundle docs.
- `inference-engine/detect-bundles/.gitkeep` — keeps the empty catalog directory tracked by git.
- `inference-engine/classify-bundles/README.md` — same shape, classify-side metrics.
- `inference-engine/classify-bundles/.gitkeep`.

### Files deleted

- `inference-engine/crl-bundles/` — entire directory: `README.md`, `multiscale-default` symlink, both bundle subdirs.
- `inference-engine/build/crl-export/` — staging dir from older build, replaced by per-pod staging dirs (build artifact, not in git).

### Files modified

- `inference-engine/protos/inference.proto` — drop `z_fused`, `z_audio`, `z_seismic` from `DetectionResult`.
- `inference-engine/inference-protos/inference_protos/inference_pb2.py` — regenerated.
- `inference-engine/src/infer_detect/main.py` — drop `z` payload writes from published `DetectionResult`.
- `inference-engine/src/infer_classify/main.py` — load encoder + type head; re-encode raw waveform from `detection.sensor_data`.
- `inference-engine/src/infer_detect/Dockerfile` — `COPY build/detect-export /app/model`; add `LABEL com.darkcircle.detect_bundle`.
- `inference-engine/src/infer_classify/Dockerfile` — `COPY build/classify-export /app/model`; add `LABEL com.darkcircle.classify_bundle`.
- `inference-engine/scripts/build_containers.sh` — replace `stage_crl_export` with `stage_bundle <kind>`; per-pod staging; hard error on `CRL_BUNDLE`.
- `crl-train/export_for_inference.py` — add `--bundle-kind {detect,classify}` (required); split `--bundles-dir` default into per-kind paths; replace `--update-default-symlink` with `--promote-default`; read metrics from `report.json`; write per-kind bundle layout.
- `inference-engine/chart/values.yaml` — add `bundles.detect` / `bundles.classify` keys (documentation-only — build-time selectors).
- `inference-engine/chart/README.md` — document the two independent build-time selectors.
- `inference-engine/README.md` — rewrite the "CRL inference deployment" section, validation-status table footnote, and `CRL_BUNDLE` references.
- `inference-engine/tests/test_infer_classify.py` — fixtures populate `sensor_data.acoustic_data` / `seismic_data` and mock the encoder forward pass.
- `inference-engine/tests/test_infer_detect.py` — fixtures drop `z_*` assertions; meta.json fixtures drop classify-only fields.
- `inference-engine/tests/test_egress.py` — `DetectionResult` fixtures drop `z_*` fields.

### Files NOT modified

- `inference-engine/src/ingestor/` — out of scope.
- `inference-engine/src/discovery/`, `egress/`, `fake_publisher/` — out of scope (egress has nothing to change; it doesn't read `z_*`).
- `inference-engine/scripts/local_smoke.sh`, `replay_in_kind.sh` — bundle selection is upstream of these.
- `inference-engine/k8s/*.yaml` — bundle selection is build-time, not k8s-runtime.
- `crl-train/training/`, `crl-train/eval.py`, `crl-train/configs/` — out of scope.

---

## Task 1: Drop `z_*` fields from `DetectionResult` proto

**Why first:** every other change cascades from this. Tests that construct `DetectionResult` need updating, and both pods need to stop reading/writing fields that no longer exist.

**Files:**
- Modify: `inference-engine/protos/inference.proto`
- Regenerate: `inference-engine/inference-protos/inference_protos/inference_pb2.py`

- [ ] **Step 1: Edit the proto**

Open `inference-engine/protos/inference.proto`. Find the `DetectionResult` message:

```proto
// Sent from Detector to Classifier (if vehicle == true) and Egress Node
message DetectionResult {
    SensorData sensor_data = 1;
    bool vehicle_detected = 2;
    float confidence = 3;
    // Latent vector(s) emitted by the CRL encoder. Only one of (z_fused) or
    // (z_audio + z_seismic) is populated, depending on the deployed model's
    // mode. The classifier reads whichever one is present and applies its
    // type head(s) without recomputing the encoder.
    repeated float z_fused = 4;
    repeated float z_audio = 5;
    repeated float z_seismic = 6;
}
```

Replace with:

```proto
// Sent from Detector to Classifier (if vehicle == true) and Egress Node.
// The classifier re-encodes the waveform from sensor_data; no latent is
// carried on the wire.
message DetectionResult {
    SensorData sensor_data = 1;
    bool vehicle_detected = 2;
    float confidence = 3;
}
```

- [ ] **Step 2: Regenerate `inference_pb2.py`**

Run from `inference-engine/`:

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run python scripts/compile_protos.py
```

Expected: prints success and rewrites `inference-protos/inference_protos/inference_pb2.py`.

- [ ] **Step 3: Verify the regenerated file no longer references `z_fused`**

Run:

```bash
grep -c 'z_fused\|z_audio\|z_seismic' inference-engine/inference-protos/inference_protos/inference_pb2.py
```

Expected output: `0`.

- [ ] **Step 4: Commit**

```bash
git add inference-engine/protos/inference.proto inference-engine/inference-protos/inference_protos/inference_pb2.py
git commit -m "proto: drop z_* fields from DetectionResult

Classify will re-encode from raw waveform after the upcoming pod refactor;
the latent no longer crosses the wire."
```

---

## Task 2: Update `tests/test_egress.py` for the proto change

**Why:** the proto change broke any test that constructs a `DetectionResult` with `z_*` fields. Fix tests before changing pods so we never run the suite from a knowingly-broken baseline.

**Files:**
- Modify: `inference-engine/tests/test_egress.py`

- [ ] **Step 1: Find every `z_fused` / `z_audio` / `z_seismic` reference in the test file**

Run:

```bash
grep -n 'z_fused\|z_audio\|z_seismic' inference-engine/tests/test_egress.py
```

Note the line numbers. For each line that builds a `DetectionResult` and sets one of these fields, the next step is to delete the assignment.

- [ ] **Step 2: Delete the `z_*` field assignments**

For each match from Step 1, locate the surrounding `DetectionResult` construction (typically a few lines that look like `result.z_fused.extend([...])` or `result.z_fused[:] = [...]`). Delete only those assignment lines. Do NOT delete the `DetectionResult` itself or its other fields.

Example before:

```python
result = inference_pb2.DetectionResult()
result.sensor_data.CopyFrom(sd)
result.vehicle_detected = True
result.confidence = 0.9
result.z_fused.extend([0.1] * 32)
```

Example after:

```python
result = inference_pb2.DetectionResult()
result.sensor_data.CopyFrom(sd)
result.vehicle_detected = True
result.confidence = 0.9
```

- [ ] **Step 3: Run the egress test suite**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run pytest tests/test_egress.py -v
```

Expected: all currently-passing tests still pass; 3 skipped (rclpy-only) remain. No `AttributeError: 'DetectionResult' object has no attribute 'z_fused'` failures.

- [ ] **Step 4: Commit**

```bash
git add inference-engine/tests/test_egress.py
git commit -m "test: drop z_* references from egress fixtures"
```

---

## Task 3: Update `tests/test_infer_detect.py` — drop `z_*` write assertions and classify-only meta fields

**Files:**
- Modify: `inference-engine/tests/test_infer_detect.py`

- [ ] **Step 1: Find `z_*` references**

```bash
grep -n 'z_fused\|z_audio\|z_seismic' inference-engine/tests/test_infer_detect.py
```

For each match:
- If it's a test asserting that `infer_detect` populates `result.z_*`, delete the assertion line.
- If it's part of a meta.json fixture, no change needed yet (covered in Step 2).

- [ ] **Step 2: Find meta.json fixtures and drop `class_names` / `probe_mode`**

```bash
grep -n 'class_names\|probe_mode' inference-engine/tests/test_infer_detect.py
```

For each match inside a meta.json dict literal that's used by the detect-side loader, delete the key. Detect bundles do not carry these fields after the restructure (see spec §2). If a test asserts on these fields directly (`assert meta["class_names"] == ...`), delete the assertion.

- [ ] **Step 3: Run the detect test suite**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run pytest tests/test_infer_detect.py -v
```

Expected: all previously-passing detect tests still pass.

- [ ] **Step 4: Commit**

```bash
git add inference-engine/tests/test_infer_detect.py
git commit -m "test: drop z_* writes and classify-only meta from detect fixtures"
```

---

## Task 4: Refactor `infer_detect/main.py` — stop writing `z_*` payloads

**Files:**
- Modify: `inference-engine/src/infer_detect/main.py:140-146`

- [ ] **Step 1: Locate the `z` extension lines**

In `inference-engine/src/infer_detect/main.py`, find lines where the published `DetectionResult` extends `z_fused` / `z_audio` / `z_seismic`. The current block (around lines 136-146):

```python
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
```

- [ ] **Step 2: Drop the latent-write lines**

Replace with:

```python
        if self.mode == "fused":
            detected, prob, _z = await loop.run_in_executor(
                None, self._infer_fused, x_audio, x_seismic
            )
        else:
            detected, prob, _z_audio, _z_seismic = await loop.run_in_executor(
                None, self._infer_per_sensor, x_audio, x_seismic
            )
```

The encoder still runs (the presence head needs the latent), but the latent stays in this pod. The variable names get an underscore prefix to make it explicit that they're discarded — the helper functions `_infer_fused` and `_infer_per_sensor` aren't changed; they keep returning the latent so a future feature could read it without another refactor.

- [ ] **Step 3: Run the detect test suite**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run pytest tests/test_infer_detect.py -v
```

Expected: pass (the assertions about `z_*` writes were already removed in Task 3).

- [ ] **Step 4: Commit**

```bash
git add inference-engine/src/infer_detect/main.py
git commit -m "infer_detect: stop writing z_* on DetectionResult

Classify re-encodes from raw waveform now; the latent is local to detect."
```

---

## Task 5: Refactor `infer_classify/main.py` — load encoder + re-encode from waveform

**Files:**
- Modify: `inference-engine/src/infer_classify/main.py`
- Reference (don't modify): `inference-engine/src/infer_detect/main.py:36-79` — same artifact-loading shape applies to classify now.

The classify pod becomes a full encode-then-classify pipeline. It loads both `encoder_*.ts` and `type_head_*.ts` from `MODEL_DIR`, reads `detection.sensor_data.acoustic_data` / `seismic_data` directly, packs to tensors using its own `meta.json`'s window sizes, runs the encoder, then runs the head.

- [ ] **Step 1: Replace the entire file with the refactored version**

Rewrite `inference-engine/src/infer_classify/main.py` to the following (the existing file's docstring + imports stay intact at the top; the loader, the inference helpers, and the `on_detection_result` callback all change):

```python
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
```

- [ ] **Step 2: Verify the file compiles cleanly**

Run:

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run python -c "import ast; ast.parse(open('src/infer_classify/main.py').read()); print('ok')"
```

Expected output: `ok`.

- [ ] **Step 3: Commit (tests come in Task 6 — they must follow the structural change)**

```bash
git add inference-engine/src/infer_classify/main.py
git commit -m "infer_classify: load encoder, re-encode from raw waveform

Classify becomes self-sufficient — it no longer depends on detect's latent.
Reads sd.acoustic_data / sd.seismic_data, packs to tensors using its own
meta.json's window sizes, runs encoder + type head."
```

---

## Task 6: Update `tests/test_infer_classify.py` for the new pod shape

**Files:**
- Modify: `inference-engine/tests/test_infer_classify.py`

The classify pod now loads two artifacts (encoder + type head) and re-encodes the raw waveform on every positive message. Tests that built a `DetectionResult` with `z_*` populated need to populate `sensor_data.acoustic_data` / `seismic_data` instead.

- [ ] **Step 1: Find all references that need updating**

```bash
grep -n 'z_fused\|z_audio\|z_seismic\|InferClassifyNode\|_infer_fused\|_infer_per_sensor' \
    inference-engine/tests/test_infer_classify.py
```

For each match:
- `z_*` field assignments on `DetectionResult` → replace with `sensor_data.acoustic_data` / `seismic_data` population.
- `InferClassifyNode(nc, heads, meta, device)` constructor calls → update to `InferClassifyNode(nc, encoders, heads, meta, device)`.
- `_infer_fused(z)` calls → update to `_infer_fused(x_audio, x_seismic)`.
- `_infer_per_sensor(z_audio, z_seismic)` calls → update to `_infer_per_sensor(x_audio, x_seismic)`.

- [ ] **Step 2: Rewrite fixture-construction code**

Anywhere a test builds a `DetectionResult` to feed into the classify pod, the new shape is:

```python
import torch
from inference_protos import inference_pb2

audio_samples = [0.1] * 16000
seismic_samples = [0.05] * 100

detection = inference_pb2.DetectionResult()
detection.sensor_data.sensor_id = "shake-001"
detection.sensor_data.acoustic_data.data.extend(audio_samples)
detection.sensor_data.acoustic_data.shape.extend([1, 16000])
detection.sensor_data.seismic_data.data.extend(seismic_samples)
detection.sensor_data.seismic_data.shape.extend([1, 100])
detection.vehicle_detected = True
detection.confidence = 0.9
```

The window sizes (16000 audio, 100 seismic) MUST match whatever `audio_window_size` / `seismic_window_size` the test's mocked `meta` dict declares — the classify pod validates lengths against its own meta.

- [ ] **Step 3: Update mock-encoder fixtures**

Anywhere the test mocks the type head module to be loaded by `_load_artifacts`, also mock an encoder. The encoder mock returns a `(z, pres_logit)` tuple of the right shape. Example with `unittest.mock`:

```python
import torch
from unittest.mock import MagicMock

# Encoder mock: returns (z, pres_logit). pres_logit is unused by classify
# but the scripted module's signature returns it, so the mock matches.
encoder_mock = MagicMock()
encoder_mock.return_value = (
    torch.randn(1, 32),       # z (batch=1, z_dim=32)
    torch.zeros(1, 1),        # pres_logit (unused here)
)

# Type-head mock: returns logits (batch=1, num_classes=4).
type_head_mock = MagicMock()
type_head_mock.return_value = torch.tensor([[0.1, 0.2, 2.5, 0.3]])
```

Patch both into `_load_artifacts` (or whatever the test's seam is), or build `InferClassifyNode` directly:

```python
encoders = {"fused": encoder_mock}
heads = {"fused": type_head_mock}
meta = {
    "mode": "fused",
    "sensors": ["audio", "seismic"],
    "class_names": ["pedestrian", "light", "medium", "heavy"],
    "z_dim": 32,
    "audio_window_size": 16000,
    "seismic_window_size": 100,
}
node = InferClassifyNode(
    nc=MagicMock(),
    encoders=encoders,
    heads=heads,
    meta=meta,
    device=torch.device("cpu"),
)
```

- [ ] **Step 4: Run the classify test suite**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run pytest tests/test_infer_classify.py -v
```

Expected: pass.

- [ ] **Step 5: Run the full suite**

```bash
poetry run pytest tests/ -v
```

Expected: 167 passed, 3 skipped (rclpy-only), 0 failed. **This is the baseline acceptance gate from the spec.** If it doesn't hit baseline, do not proceed — fix the regression in this task before moving on.

- [ ] **Step 6: Commit**

```bash
git add inference-engine/tests/test_infer_classify.py
git commit -m "test: adapt classify fixtures to encoder+head pod shape

Tests now populate sensor_data on DetectionResult and mock the encoder
forward pass alongside the type head."
```

---

## Task 7: Switch detect pod's Dockerfile to `build/detect-export`

**Files:**
- Modify: `inference-engine/src/infer_detect/Dockerfile`

- [ ] **Step 1: Update the COPY line and add a label**

Find the line in `inference-engine/src/infer_detect/Dockerfile`:

```dockerfile
# CRL TorchScript bundle (encoder_*.ts + meta.json). Staged into
# build/crl-export/ by scripts/build_containers.sh from a saved CRL run.
# Override at runtime by mounting a different directory at /app/model.
COPY build/crl-export /app/model
```

Replace with:

```dockerfile
# CRL TorchScript bundle (encoder_*.ts + meta.json). Staged into
# build/detect-export/ by scripts/build_containers.sh from the bundle
# named by DETECT_BUNDLE (default: detect-default).
# Override at runtime by mounting a different directory at /app/model.
ARG DETECT_BUNDLE_LABEL=unknown
LABEL com.darkcircle.detect_bundle=$DETECT_BUNDLE_LABEL
COPY build/detect-export /app/model
```

- [ ] **Step 2: Verify the file is syntactically OK**

Run:

```bash
docker buildx build --help >/dev/null 2>&1 && echo "docker on PATH"
```

(No build yet — the build script needs Task 9's changes first to populate `build/detect-export/`. We'll exercise the Dockerfile end-to-end in Task 13.)

- [ ] **Step 3: Commit**

```bash
git add inference-engine/src/infer_detect/Dockerfile
git commit -m "infer_detect: COPY build/detect-export; label bundle name"
```

---

## Task 8: Switch classify pod's Dockerfile to `build/classify-export`

**Files:**
- Modify: `inference-engine/src/infer_classify/Dockerfile`

- [ ] **Step 1: Update the COPY line and add a label**

Find the line in `inference-engine/src/infer_classify/Dockerfile`:

```dockerfile
# CRL TorchScript bundle (type_head_*.ts + meta.json). Staged into
# build/crl-export/ by scripts/build_containers.sh.
COPY build/crl-export /app/model
```

Replace with:

```dockerfile
# CRL TorchScript bundle (encoder_*.ts + type_head_*.ts + meta.json).
# Staged into build/classify-export/ by scripts/build_containers.sh from
# the bundle named by CLASSIFY_BUNDLE (default: classify-default).
ARG CLASSIFY_BUNDLE_LABEL=unknown
LABEL com.darkcircle.classify_bundle=$CLASSIFY_BUNDLE_LABEL
COPY build/classify-export /app/model
```

- [ ] **Step 2: Commit**

```bash
git add inference-engine/src/infer_classify/Dockerfile
git commit -m "infer_classify: COPY build/classify-export; label bundle name"
```

---

## Task 9: Rewrite `scripts/build_containers.sh` for per-pod bundle staging

**Files:**
- Modify: `inference-engine/scripts/build_containers.sh`

The script needs to:
1. Read `DETECT_BUNDLE` / `CLASSIFY_BUNDLE` (defaulting to `detect-default` / `classify-default`).
2. Read `DETECT_RUN_DIR` / `CLASSIFY_RUN_DIR` for the dev re-export fallback (split from today's `CRL_RUN_DIR`).
3. Stage each pod's bundle into its own directory (`build/detect-export/`, `build/classify-export/`).
4. Pass `<KIND>_BUNDLE_LABEL` as a build-arg to `docker build` so the Dockerfile's `LABEL` gets the bundle name.
5. Hard-error if `CRL_BUNDLE` is set in the environment.

- [ ] **Step 1: Replace the file with the rewritten version**

Open `inference-engine/scripts/build_containers.sh` and replace its contents with:

```bash
#!/usr/bin/env bash
# Build container images and (optionally) load them into a kind cluster
# or push them to a registry.
#
# Build context is the inference-engine root for every image so that
# Dockerfiles can COPY src/<node>/, k8s/, ros2_interfaces/, and
# inference-protos/ without escaping the context.
#
# Usage:
#   # Build all images, default registry+tag (inference-engine/*:dev),
#   # auto-load into the dark-circle kind cluster if it exists.
#   scripts/build_containers.sh
#
#   # Subset
#   scripts/build_containers.sh discovery egress
#
#   # Build for a specific kind cluster
#   KIND_CLUSTER=mycluster scripts/build_containers.sh
#
#   # Build, tag, and push to a remote registry (Helm-chart workflow)
#   REGISTRY=registry.example.com/dark-circle TAG=v0.1.0 PUSH=1 \
#       scripts/build_containers.sh
#
#   # Pick specific bundles for the inference pods
#   DETECT_BUNDLE=multiscale-vae-<run>-v1 \
#   CLASSIFY_BUNDLE=multiscale-vae-<run>-mlp_ztype-v1 \
#       scripts/build_containers.sh infer-detect infer-classify
#
# Behavior:
#   - kind load: auto-runs when ``kind`` is on PATH AND the cluster
#     ``$KIND_CLUSTER`` (default ``dark-circle``) exists. Skipped silently
#     otherwise — useful for build-only and customer-registry workflows.
#   - docker push: only runs when ``PUSH=1`` is set. Caller is responsible
#     for ``docker login`` against the target registry beforehand.
set -eo pipefail

# Hard error on the retired single-bundle env var. Silent fallthrough
# during the restructure would be a footgun.
if [ -n "${CRL_BUNDLE:-}" ]; then
    echo "ERROR: CRL_BUNDLE is no longer supported." >&2
    echo "Use DETECT_BUNDLE and CLASSIFY_BUNDLE instead." >&2
    echo "  DETECT_BUNDLE=<name>   selects from inference-engine/detect-bundles/" >&2
    echo "  CLASSIFY_BUNDLE=<name> selects from inference-engine/classify-bundles/" >&2
    exit 1
fi
if [ -n "${CRL_RUN_DIR:-}" ]; then
    echo "ERROR: CRL_RUN_DIR is no longer supported." >&2
    echo "Use DETECT_RUN_DIR and CLASSIFY_RUN_DIR instead (dev fallback path)." >&2
    exit 1
fi

REGISTRY="${REGISTRY:-inference-engine}"
TAG="${TAG:-dev}"
KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# Per-pod bundle selectors. Each pod is staged independently.
DETECT_BUNDLE="${DETECT_BUNDLE:-detect-default}"
CLASSIFY_BUNDLE="${CLASSIFY_BUNDLE:-classify-default}"

# Dev re-export fallback. Used when <KIND>_BUNDLE doesn't resolve to a
# bundle dir AND crl-train is installed alongside.
DETECT_RUN_DIR="${DETECT_RUN_DIR:-}"
CLASSIFY_RUN_DIR="${CLASSIFY_RUN_DIR:-}"
CRL_TRAIN_PYTHON="${CRL_TRAIN_PYTHON:-$REPO_ROOT/crl-train/.venv/bin/python}"

# stage_bundle <kind>
#
# kind = "detect" or "classify". Stages the corresponding bundle into
# build/<kind>-export/ so the matching Dockerfile can COPY it. Resolution
# order: pre-bundled (<kind>-bundles/$<KIND>_BUNDLE) first, then re-export
# from $<KIND>_RUN_DIR via crl-train.
stage_bundle() {
    local kind="$1"
    local bundle_var run_dir_var bundle_name run_dir bundle_path out stamp resolved bundle_id

    case "$kind" in
        detect)
            bundle_name="$DETECT_BUNDLE"
            run_dir="$DETECT_RUN_DIR"
            ;;
        classify)
            bundle_name="$CLASSIFY_BUNDLE"
            run_dir="$CLASSIFY_RUN_DIR"
            ;;
        *)
            echo "stage_bundle: unknown kind '$kind' (expected detect or classify)" >&2
            return 1
            ;;
    esac

    bundle_path="$ROOT/${kind}-bundles/$bundle_name"
    out="$ROOT/build/${kind}-export"
    stamp="$out/.bundle_id"

    # ---- Path 1: pre-bundled artifact ---------------------------------
    if [ -d "$bundle_path" ]; then
        resolved="$(cd "$bundle_path" && pwd -P)"
        bundle_id="$(basename "$resolved")"

        if [ -f "$stamp" ] && \
           [ "$(cat "$stamp" 2>/dev/null)" = "$bundle_id" ] && \
           [ -z "${CRL_FORCE_EXPORT:-}" ]; then
            echo "=== ${kind} bundle '$bundle_id' already staged at $out (set CRL_FORCE_EXPORT=1 to redo) ==="
            return 0
        fi

        echo "=== copying ${kind} bundle '$bundle_id' -> build/${kind}-export ==="
        rm -rf "$out"
        mkdir -p "$out"
        cp -RL "$bundle_path"/. "$out"/
        printf "%s\n" "$bundle_id" > "$stamp"
        return 0
    fi

    # ---- Path 2: re-export from a crl-train saved run -----------------
    if [ -z "$run_dir" ] || [ ! -d "$REPO_ROOT/$run_dir" ]; then
        echo "${kind} bundle '$bundle_name' not found at $bundle_path" >&2
        if [ -z "$run_dir" ]; then
            echo "and ${kind^^}_RUN_DIR is not set." >&2
        else
            echo "and ${kind^^}_RUN_DIR not found at $REPO_ROOT/$run_dir" >&2
        fi
        echo "" >&2
        echo "Set ${kind^^}_BUNDLE to a directory under ${kind}-bundles/," >&2
        echo "or install crl-train and point ${kind^^}_RUN_DIR at a saved run." >&2
        exit 1
    fi
    if [ ! -x "$CRL_TRAIN_PYTHON" ]; then
        echo "${kind} bundle '$bundle_name' not found at $bundle_path" >&2
        echo "and CRL_TRAIN_PYTHON not executable: $CRL_TRAIN_PYTHON" >&2
        echo "" >&2
        echo "Customer path: set ${kind^^}_BUNDLE to a bundle under ${kind}-bundles/." >&2
        echo "Dev path:      cd $REPO_ROOT/crl-train && poetry install" >&2
        exit 1
    fi

    local run_id="run:$kind:$(basename "$run_dir")"
    if [ -f "$stamp" ] && \
       [ "$(cat "$stamp" 2>/dev/null)" = "$run_id" ] && \
       [ -z "${CRL_FORCE_EXPORT:-}" ]; then
        echo "=== ${kind} re-export for '$run_id' already staged (set CRL_FORCE_EXPORT=1 to redo) ==="
        return 0
    fi

    rm -rf "$out"
    mkdir -p "$out"
    echo "=== re-exporting ${kind} from $run_dir -> build/${kind}-export ==="
    (cd "$REPO_ROOT/crl-train" && \
        "$CRL_TRAIN_PYTHON" export_for_inference.py \
            --save-dir "$REPO_ROOT/$run_dir" \
            --bundle-kind "$kind" \
            --out-dir "$out")
    printf "%s\n" "$run_id" > "$stamp"
}

# Image short-name -> Dockerfile path lookup.
dockerfile_for() {
    case "$1" in
        discovery)      echo "src/discovery/Dockerfile" ;;
        ingestor)       echo "src/ingestor/Dockerfile" ;;
        fake-publisher) echo "src/fake_publisher/Dockerfile" ;;
        infer-detect)   echo "src/infer_detect/Dockerfile" ;;
        infer-classify) echo "src/infer_classify/Dockerfile" ;;
        egress)         echo "src/egress/Dockerfile" ;;
        *)              return 1 ;;
    esac
}

ALL_TARGETS="discovery ingestor fake-publisher infer-detect infer-classify egress"

if [ "$#" -gt 0 ]; then
    TARGETS="$*"
else
    TARGETS="$ALL_TARGETS"
fi

for name in $TARGETS; do
    dockerfile=$(dockerfile_for "$name") || {
        echo "unknown image: $name (valid: $ALL_TARGETS)" >&2
        exit 1
    }

    # Per-pod bundle staging + label build-arg.
    build_args=()
    case "$name" in
        infer-detect)
            stage_bundle detect
            build_args+=(--build-arg "DETECT_BUNDLE_LABEL=$DETECT_BUNDLE")
            ;;
        infer-classify)
            stage_bundle classify
            build_args+=(--build-arg "CLASSIFY_BUNDLE_LABEL=$CLASSIFY_BUNDLE")
            ;;
    esac

    image="${REGISTRY}/${name}:${TAG}"
    echo "=== building $image ==="
    docker build "${build_args[@]}" -f "$dockerfile" -t "$image" .
done

if command -v kind >/dev/null 2>&1; then
    if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER"; then
        for name in $TARGETS; do
            image="${REGISTRY}/${name}:${TAG}"
            echo "=== loading $image into kind cluster '$KIND_CLUSTER' ==="
            kind load docker-image "$image" --name "$KIND_CLUSTER"
        done
    else
        echo "kind cluster '$KIND_CLUSTER' not found; skipping kind load" >&2
        echo "(create it with: kind create cluster --name $KIND_CLUSTER)" >&2
    fi
else
    echo "kind not on PATH; skipping kind load" >&2
fi

if [ "${PUSH:-}" = "1" ]; then
    for name in $TARGETS; do
        image="${REGISTRY}/${name}:${TAG}"
        echo "=== pushing $image ==="
        docker push "$image"
    done
fi
```

- [ ] **Step 2: Make sure it's still executable**

```bash
chmod +x inference-engine/scripts/build_containers.sh
```

- [ ] **Step 3: Smoke-test the hard-error path**

```bash
CRL_BUNDLE=foo bash inference-engine/scripts/build_containers.sh infer-detect 2>&1 | head -5
```

Expected: prints the `ERROR: CRL_BUNDLE is no longer supported.` block and exits non-zero.

```bash
echo "exit: $?"
```

Should be `1`.

- [ ] **Step 4: Smoke-test the missing-bundle path**

```bash
DETECT_BUNDLE=does-not-exist bash inference-engine/scripts/build_containers.sh infer-detect 2>&1 | head -10
```

Expected: prints `detect bundle 'does-not-exist' not found at .../detect-bundles/does-not-exist`, then the run_dir check fails too (since `DETECT_RUN_DIR` is unset), exits non-zero.

- [ ] **Step 5: Commit**

```bash
git add inference-engine/scripts/build_containers.sh
git commit -m "build_containers: stage detect+classify bundles independently

Splits stage_crl_export into stage_bundle <kind>; per-pod bundle env vars
(DETECT_BUNDLE / CLASSIFY_BUNDLE) and per-pod run-dir fallbacks. Hard-errors
on the retired CRL_BUNDLE / CRL_RUN_DIR variables."
```

---

## Task 10: Add `--bundle-kind` flag to `crl-train/export_for_inference.py`

**Files:**
- Modify: `crl-train/export_for_inference.py`

The exporter changes:
1. New required flag `--bundle-kind {detect,classify}` whenever `--bundle-name` or `--out-dir` is used.
2. `--bundles-dir` default splits per-kind: `detect-bundles/` for `--bundle-kind detect`, `classify-bundles/` for `--bundle-kind classify`. The single `crl-bundles/` default is removed.
3. The bundle layout written to disk is per-kind:
   - `detect`: encoder `.ts` files + `meta.json` (no type head, no `class_names` / `probe_mode`).
   - `classify`: encoder `.ts` files + type head `.ts` files + `meta.json` (with `class_names` / `probe_mode`).
4. `meta.json` carries selection metrics read from the source run's `report.json`:
   - detect: `pres_f1`, `min_pres_f1`, `source_run`.
   - classify: `type_f1`, `min_type_f1`, `source_run`.

The implementation is split into multiple steps so each piece is independently reviewable.

- [ ] **Step 1: Add per-kind bundle parent paths and update `_BUNDLE_NAME_RE`**

In `crl-train/export_for_inference.py` near the top (after the existing `_DEFAULT_BUNDLES_PARENT` line, around line 71), replace:

```python
_DEFAULT_BUNDLES_PARENT = Path(__file__).resolve().parent.parent / "inference-engine" / "crl-bundles"
```

With:

```python
_INFERENCE_ENGINE = Path(__file__).resolve().parent.parent / "inference-engine"
_DEFAULT_DETECT_BUNDLES_PARENT = _INFERENCE_ENGINE / "detect-bundles"
_DEFAULT_CLASSIFY_BUNDLES_PARENT = _INFERENCE_ENGINE / "classify-bundles"
```

Leave `_BUNDLE_NAME_RE` unchanged — the `-v<N>` suffix rule applies to both kinds.

- [ ] **Step 2: Add a `report.json` reader for selection metrics**

Add this helper near `build_deployment_meta` (around line 800):

```python
def _read_selection_metrics(save_dir: Path, kind: str) -> dict:
    """Pull the metric fields the bundle catalog needs from report.json.

    detect bundles need `pres_f1` (selection) + `min_pres_f1` (tie-breaker).
    classify bundles need `type_f1` (selection) + `min_type_f1`
    (selection floor + tie-breaker). Missing fields are a hard error.
    """
    report_path = save_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"report.json not found in {save_dir} — needed for bundle "
            f"selection metrics. Re-run eval to produce it."
        )
    report = json.loads(report_path.read_text())

    if kind == "detect":
        required = ("pres_f1", "min_pres_f1")
    elif kind == "classify":
        required = ("type_f1", "min_type_f1")
    else:
        raise ValueError(f"unknown bundle kind: {kind!r}")

    out: dict = {}
    for field in required:
        if field not in report:
            raise KeyError(
                f"report.json in {save_dir} is missing required field "
                f"{field!r} for kind={kind}. Available keys: "
                f"{sorted(report.keys())}"
            )
        out[field] = float(report[field])
    out["source_run"] = save_dir.name
    return out
```

- [ ] **Step 3: Split `build_deployment_meta` into per-kind variants**

Replace the existing `build_deployment_meta` function (around line 802-829) with:

```python
def build_deployment_meta(
    cfg: CRLConfig,
    sensors: list[str],
    mode: str,
    presence_threshold: dict | float,
    probe_mode: str,
    kind: str,
    selection_metrics: dict,
) -> dict:
    """Subset of training meta.json that the inference pods need.

    Inference pods only need: which mode, which sensors, what shapes the
    encoder expects, the threshold (detect-only), the class names
    (classify-only), the probe (classify-only), the latent dim, and the
    selection metrics for catalog ranking.
    """
    meta: dict = {
        "frontend_type": cfg.frontend_type,
        "mode": mode,  # "per_sensor" | "fused"
        "sensors": sensors,
        "z_dim": cfg.d_z,
    }
    if kind == "detect":
        meta["presence_threshold"] = presence_threshold
    elif kind == "classify":
        meta["class_names"] = CLASS_NAMES
        meta["probe_mode"] = probe_mode
    else:
        raise ValueError(f"unknown bundle kind: {kind!r}")

    for sensor in sensors:
        mc = cfg.modality_cfg(sensor)
        meta[f"{sensor}_sample_rate"] = mc.sample_rate
        meta[f"{sensor}_window_size"] = mc.window_size

    meta.update(selection_metrics)
    return meta
```

- [ ] **Step 4: Add `--bundle-kind` to `parse_args`**

In `parse_args` (around line 837), add a required `--bundle-kind` argument right after the `--save-dir` argument (around line 845):

```python
    ap.add_argument(
        "--bundle-kind",
        choices=["detect", "classify"],
        required=True,
        help=(
            "Which kind of bundle to write. "
            "'detect' = encoder + presence-only meta (deployed to infer-detect). "
            "'classify' = encoder + type head + classify meta (deployed to infer-classify). "
            "A single saved run produces one bundle per kind via two separate "
            "invocations."
        ),
    )
```

Also add `--promote-default` (replacing `--update-default-symlink`) — find the existing `--update-default-symlink` block (around line 877-884) and replace with:

```python
    ap.add_argument(
        "--promote-default",
        action="store_true",
        help=(
            "After a successful --bundle-name export, walk the catalog for "
            "this --bundle-kind, apply the selection rules, and repoint "
            "<kind>-default at the winner if it changed. Only valid with "
            "--bundle-name."
        ),
    )
```

Then later in `parse_args` where the validation happens (around line 937-938):

```python
    if args.update_default_symlink and args.bundle_name is None:
        ap.error("--update-default-symlink requires --bundle-name")
```

Replace with:

```python
    if args.promote_default and args.bundle_name is None:
        ap.error("--promote-default requires --bundle-name")
```

- [ ] **Step 5: Update `--bundles-dir` default to be kind-aware**

Find the `--bundles-dir` argument (around line 868-875) and replace with:

```python
    ap.add_argument(
        "--bundles-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory holding bundle subdirs. Defaults to "
            "<inference-engine>/<kind>-bundles/ per --bundle-kind."
        ),
    )
```

Then in `parse_args` resolution (after the `--bundle-kind` is parsed but before `--bundle-name` is resolved to an `out_dir`), add a default-resolution step. Find the block (around line 921-935) starting `if args.bundle_name is not None:` and replace with:

```python
    # Resolve --bundles-dir per kind if not supplied.
    if args.bundles_dir is None:
        if args.bundle_kind == "detect":
            args.bundles_dir = _DEFAULT_DETECT_BUNDLES_PARENT
        else:
            args.bundles_dir = _DEFAULT_CLASSIFY_BUNDLES_PARENT

    # Resolve --bundle-name to a concrete out_dir, with validation.
    if args.bundle_name is not None:
        if not _BUNDLE_NAME_RE.match(args.bundle_name):
            ap.error(
                f"--bundle-name {args.bundle_name!r} doesn't match the convention "
                f"<name>-v<N>. See inference-engine/<kind>-bundles/README.md for "
                f"the full naming guide."
            )
        if not args.bundles_dir.is_dir():
            ap.error(
                f"--bundles-dir {args.bundles_dir} does not exist. "
                f"Pass --bundles-dir explicitly or place crl-train and "
                f"inference-engine as siblings under one parent dir."
            )
        args.out_dir = args.bundles_dir / args.bundle_name
```

- [ ] **Step 6: Adapt `main()` to write per-kind layout**

In `main()`, find the per-sensor branch (around line 957-982) and the fused branch (around 984-1006). The current code always writes both `encoder_*.ts` AND `type_head_*.ts`. Wrap the type-head writes in `if args.bundle_kind == "classify":` so detect bundles skip them.

Replace the per-sensor branch:

```python
    if cfg.frontend_type in PER_SENSOR_FRONTENDS:
        mode = "per_sensor"
        threshold_dict = {
            "audio": float(args.threshold_audio),
            "seismic": float(args.threshold_seismic),
        }
        presence_threshold: dict | float = threshold_dict

        for sensor in sensors:
            mc = cfg.modality_cfg(sensor)
            print(f"\nExporting per-sensor [{sensor}] (window={mc.window_size})")
            enc_eager, type_eager = build_per_sensor_wrappers(model, sensor, type_slice)
            enc_path = args.out_dir / f"encoder_{sensor}.ts"
            scripted_enc = script_and_save(enc_eager, enc_path)
            print(f"  wrote {enc_path.name}")

            scripted_type = None
            if args.bundle_kind == "classify":
                type_path = args.out_dir / f"type_head_{sensor}.ts"
                scripted_type = script_and_save(type_eager, type_path)
                print(f"  wrote {type_path.name}")

            if not args.skip_parity and scripted_type is not None:
                parity_check_per_sensor(
                    enc_eager,
                    type_eager,
                    scripted_enc,
                    scripted_type,
                    window_size=mc.window_size,
                )
                print("  parity OK (atol=1e-5)")
```

Replace the fused branch:

```python
    elif cfg.frontend_type in FUSED_FRONTENDS:
        mode = "fused"
        presence_threshold = float(args.threshold_fused)

        print("\nExporting fused encoder (audio + seismic)")
        enc_eager, type_eager = build_fused_wrappers(model, type_slice)
        enc_path = args.out_dir / "encoder_fused.ts"
        scripted_enc = script_and_save(enc_eager, enc_path)
        print(f"  wrote {enc_path.name}")

        scripted_type = None
        if args.bundle_kind == "classify":
            type_path = args.out_dir / "type_head_fused.ts"
            scripted_type = script_and_save(type_eager, type_path)
            print(f"  wrote {type_path.name}")

        if not args.skip_parity and scripted_type is not None:
            audio_window = cfg.modality_cfg("audio").window_size
            seismic_window = cfg.modality_cfg("seismic").window_size
            parity_check_fused(
                enc_eager,
                type_eager,
                scripted_enc,
                scripted_type,
                audio_window=audio_window,
                seismic_window=seismic_window,
            )
            print("  parity OK (atol=1e-5)")
```

(Detect bundles skip parity because they don't write a type head; only the encoder is parity-checked, and that's covered by classify export when it runs against the same source run.)

Then update the meta-write block at the bottom of `main()` (around line 1015-1024) to call the new signature:

```python
    selection_metrics = _read_selection_metrics(args.save_dir, args.bundle_kind)

    deploy_meta = build_deployment_meta(
        cfg=cfg,
        sensors=sensors,
        mode=mode,
        presence_threshold=presence_threshold,
        probe_mode=probe_mode,
        kind=args.bundle_kind,
        selection_metrics=selection_metrics,
    )
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(deploy_meta, indent=2) + "\n")
    print(f"\nWrote {meta_path}")
    print(f"Done. {args.out_dir}/ ready for inference-engine deploy.")
```

Finally update the symlink-promotion block (around line 1027-1046). Replace:

```python
    if args.update_default_symlink:
        link_path = args.bundles_dir / "multiscale-default"
        target = args.bundle_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target)
        print(f"Repointed {link_path} -> {target}")
        print(
            "Remember to update the catalog table in "
            "inference-engine/crl-bundles/README.md and commit."
        )
    elif args.bundle_name is not None:
        print(
            f"\nTo make this the new shipping default, re-run with "
            f"--update-default-symlink, or manually:\n"
            f"  ln -sfn {args.bundle_name} "
            f"{args.bundles_dir}/multiscale-default"
        )
```

With:

```python
    if args.promote_default:
        _promote_default(args.bundles_dir, args.bundle_kind)
    elif args.bundle_name is not None:
        link_name = f"{args.bundle_kind}-default"
        print(
            f"\nTo evaluate this against the catalog and promote if it wins, "
            f"re-run with --promote-default. Or manually:\n"
            f"  ln -sfn {args.bundle_name} "
            f"{args.bundles_dir}/{link_name}"
        )
```

- [ ] **Step 7: Add the `_promote_default` function**

Add near the top of the file alongside `_read_selection_metrics`:

```python
_TIE_EPSILON = 0.01

# Floors for *-default symlink promotion. Floors gate ONLY auto-promotion;
# bundles below the floor still exist on disk and can be selected
# explicitly via DETECT_BUNDLE / CLASSIFY_BUNDLE.
_PROMOTION_FLOOR = {
    "detect": ("pres_f1", 0.80),
    "classify": ("min_type_f1", 0.40),
}

# (primary_metric, tiebreaker_metric) per kind.
_RANKING_METRICS = {
    "detect": ("pres_f1", "min_pres_f1"),
    "classify": ("type_f1", "min_type_f1"),
}


def _list_catalog(bundles_dir: Path) -> list[Path]:
    """Return every bundle subdir in bundles_dir, ignoring symlinks
    (the *-default symlink itself is one of those) and non-dirs."""
    out: list[Path] = []
    for child in sorted(bundles_dir.iterdir()):
        if child.is_dir() and not child.is_symlink():
            out.append(child)
    return out


def _rank_bundles(catalog: list[Path], kind: str) -> list[tuple[Path, dict]]:
    """Rank catalog by (primary, tiebreaker) descending. Bundles missing
    metrics are skipped with a warning."""
    primary, tiebreaker = _RANKING_METRICS[kind]
    scored: list[tuple[float, float, str, Path, dict]] = []
    for bundle in catalog:
        meta_path = bundle / "meta.json"
        if not meta_path.exists():
            print(f"  skipping {bundle.name}: no meta.json", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        if primary not in meta or tiebreaker not in meta:
            print(
                f"  skipping {bundle.name}: missing {primary!r} or {tiebreaker!r}",
                flush=True,
            )
            continue
        scored.append((float(meta[primary]), float(meta[tiebreaker]), bundle.name, bundle, meta))
    # Sort by (primary desc, tiebreaker desc, name asc). The name-asc
    # last component makes ties deterministic.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [(b, m) for _, _, _, b, m in scored]


def _promote_default(bundles_dir: Path, kind: str) -> None:
    """Re-evaluate the catalog and repoint <kind>-default at the winner
    if any bundle clears the floor."""
    floor_metric, floor_value = _PROMOTION_FLOOR[kind]
    primary, _tiebreaker = _RANKING_METRICS[kind]
    link_name = f"{kind}-default"
    link_path = bundles_dir / link_name

    catalog = _list_catalog(bundles_dir)
    if not catalog:
        print(f"  no bundles in {bundles_dir}; cannot promote {link_name}", flush=True)
        raise SystemExit(1)

    ranked = _rank_bundles(catalog, kind)
    if not ranked:
        print(f"  no bundles with valid metrics in {bundles_dir}", flush=True)
        raise SystemExit(1)

    eligible = [(b, m) for b, m in ranked if m.get(floor_metric, 0.0) >= floor_value]
    if not eligible:
        best_bundle, best_meta = ranked[0]
        print(
            f"  no eligible bundle for {link_name} "
            f"(highest {floor_metric}={best_meta.get(floor_metric, 0.0):.3f}, "
            f"floor={floor_value:.2f})",
            flush=True,
        )
        raise SystemExit(1)

    winner_bundle, winner_meta = eligible[0]
    target = winner_bundle.name

    current_target: str | None = None
    if link_path.is_symlink():
        current_target = os.readlink(link_path)

    if current_target == target:
        print(
            f"  {link_name} already points at {target} "
            f"({primary}={winner_meta[primary]:.3f}); no change",
            flush=True,
        )
        return

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    print(
        f"  repointed {link_path} -> {target} "
        f"({primary}={winner_meta[primary]:.3f})",
        flush=True,
    )
```

Add `import os` at the top of the file if not already imported.

- [ ] **Step 8: Update the docstring**

Replace the file's module docstring (lines 1-51) with one reflecting the new flag set:

```python
"""Export a trained CRLModel into TorchScript artifacts for the
inference-engine deployment pipeline.

Reads a saved run and emits per-mode TorchScript files plus a deployment-only
meta.json that the inference pods consume. The inference image has zero Python
dependency on crl_vehicle — only torch.jit.load is needed.

Two checkpoints are required (the heads are checkpointed independently):
  * downstream_best_pres.pth — argmax val_pres_f1 over training epochs
  * downstream_best_type.pth — argmax val_type_f1 over training epochs

Presence-side parameters (encoder + pres_heads + aux_pres_heads) are taken from
the pres ckpt; type-side parameters (type_heads + aux_type_heads) are taken
from the type ckpt. The encoder is shared across both heads at inference time,
so the source for it is configurable via --encoder-from (default: pres).

Bundle kinds (--bundle-kind):
  * detect   -> encoder_*.ts + meta.json (no type head). Catalog: detect-bundles/.
              meta.json adds: pres_f1, min_pres_f1, source_run.
  * classify -> encoder_*.ts + type_head_*.ts + meta.json. Catalog: classify-bundles/.
              meta.json adds: class_names, probe_mode, type_f1, min_type_f1, source_run.

A single saved CRL run produces one detect bundle and one classify bundle via
two separate invocations.

Output layout (in --out-dir):

  Per-sensor mode (frontend_type ∈ {morlet, morlet_per_sensor, morlet_learnable}):
    encoder_audio.ts        # (x_audio[B,1,16000])    -> (z[B,d_z], pres_logit[B,1])
    encoder_seismic.ts      # (x_seismic[B,1,100])    -> (z[B,d_z], pres_logit[B,1])
    type_head_audio.ts      # classify only
    type_head_seismic.ts    # classify only
    meta.json

  Fused mode (frontend_type ∈ {multiscale, morlet_fused, morlet_learnable_fused}):
    encoder_fused.ts        # (x_audio, x_seismic) -> (z, pres_logit)
    type_head_fused.ts      # classify only
    meta.json

CLI:
    # Produce a detect bundle, evaluate against the catalog, promote if winner.
    python export_for_inference.py \\
        --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \\
        --bundle-kind detect \\
        --bundle-name <frontend>-<mode>-<run>-v1 \\
        --promote-default

    # Produce a classify bundle from the same run.
    python export_for_inference.py \\
        --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \\
        --bundle-kind classify \\
        --bundle-name <frontend>-<mode>-<run>-<probe>-v1 \\
        --promote-default

    # Escape hatch: explicit out-dir for one-off exports outside the catalog.
    python export_for_inference.py --save-dir saved_crl/runs/<run> \\
        --bundle-kind detect --out-dir /tmp/scratch-bundle
"""
```

- [ ] **Step 9: Verify the file parses**

Run:

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/crl-train
poetry run python -c "import ast; ast.parse(open('export_for_inference.py').read()); print('ok')"
```

Expected output: `ok`.

- [ ] **Step 10: Verify `--help` works and lists the new flags**

```bash
poetry run python export_for_inference.py --help 2>&1 | grep -E -- '--bundle-kind|--promote-default'
```

Expected: both flags appear in the output.

- [ ] **Step 11: Commit**

```bash
git add crl-train/export_for_inference.py
git commit -m "export_for_inference: add --bundle-kind, --promote-default

--bundle-kind {detect,classify} drives the bundle layout: detect bundles
get encoder + presence-only meta; classify bundles get encoder + type head
+ class_names + probe_mode. Selection metrics (pres_f1/min_pres_f1 for
detect, type_f1/min_type_f1 for classify) are read from report.json and
written into meta.json. --promote-default re-evaluates the catalog
against the rules in the design spec and repoints <kind>-default."
```

---

## Task 11: Delete `crl-bundles/` and create the new catalog skeletons

**Files:**
- Create: `inference-engine/detect-bundles/.gitkeep`
- Create: `inference-engine/detect-bundles/README.md`
- Create: `inference-engine/classify-bundles/.gitkeep`
- Create: `inference-engine/classify-bundles/README.md`
- Delete: `inference-engine/crl-bundles/` (entire directory)

- [ ] **Step 1: Create the empty catalog directories with .gitkeep**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
mkdir -p detect-bundles classify-bundles
touch detect-bundles/.gitkeep classify-bundles/.gitkeep
```

- [ ] **Step 2: Write `detect-bundles/README.md`**

Content:

```markdown
# Detect-side CRL Bundles

Pre-exported TorchScript bundles for `infer-detect`. Each bundle is a
self-contained directory:

```
<bundle-name>/
├── meta.json              # frontend_type, mode, sensors, sample rates,
│                          #   window sizes, z_dim, presence_threshold,
│                          #   source_run, pres_f1, min_pres_f1
└── encoder_fused.ts       # fused mode: (x_audio, x_seismic) -> (z, pres_logit)
```

Per-sensor mode bundles instead carry `encoder_audio.ts` +
`encoder_seismic.ts`.

Detect bundles do **not** carry the type head — that lives in the
classify bundle.

## Selecting a bundle at build time

`scripts/build_containers.sh` reads `DETECT_BUNDLE` to pick which
directory to copy into the `infer-detect` container. Default is
`detect-default` (a symlink, not committed yet — see Bootstrap below).

```bash
# Use the default
bash scripts/build_containers.sh infer-detect

# Pick a specific bundle
DETECT_BUNDLE=multiscale-vae-<run>-v1 \
    bash scripts/build_containers.sh infer-detect

# Force a re-copy even if build/detect-export/ already has the same bundle
CRL_FORCE_EXPORT=1 bash scripts/build_containers.sh infer-detect
```

If `DETECT_BUNDLE` doesn't resolve to a directory and
`DETECT_RUN_DIR` is set, the build falls back to re-exporting from a
saved CRL run via `crl-train`'s exporter. Customer path: always use
a pre-bundled artifact.

## Naming convention

`<frontend>-<training-mode>-<run-id>-v<N>`

Example: `multiscale-vae-2026_04_29_13_26_17-v1`

- frontend: `multiscale`, `morlet`, `morlet_per_sensor`, `morlet_fused`,
  `morlet_learnable`, `morlet_learnable_fused`
- training-mode: `vae`, `contrastive`, `disentangled`
- run-id: the saved-run directory name
- v<N>: monotonic version, bump when re-exporting from a different
  checkpoint of the same run

Detect bundles **do not** include a probe component in the name — the
probe is a classify-side concept.

## Default symlink

`detect-default` points at the current shipping leader. Promotion is
driven by `--promote-default` on `export_for_inference.py`, which
re-evaluates the catalog against the selection rules below.

## Selection rules

| Metric | Role | Value |
|--------|------|-------|
| `pres_f1` | Primary | Highest on held-out eval (val) split |
| `min_pres_f1` | Tie-breaker | Highest worst-location presence F1 |
| `pres_f1` | Promotion floor | `≥ 0.80` to be eligible for `detect-default` |

Tie band: `|pres_f1_a − pres_f1_b| < 0.01`. Inside the band the
tie-breaker fires; outside it the primary alone decides.

If no bundle clears the floor, `--promote-default` exits non-zero and
leaves the symlink untouched. Bundles below the floor still exist
on disk and can be selected explicitly via `DETECT_BUNDLE=<name>`.

## Producing a new bundle

Customers don't run this. The exporter requires `crl-train` and its
saved-run directory.

```bash
cd crl-train
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind detect \
    --bundle-name <frontend>-<mode>-<run>-v1 \
    --promote-default
```

`--bundle-name` writes into `inference-engine/detect-bundles/<name>/`.
`--promote-default` re-evaluates the catalog and repoints
`detect-default` if the new bundle wins.

After exporting, commit the bundle directory and (if you promoted)
the symlink change. Update the catalog table below by hand.

## Bootstrap

This catalog is empty after the restructure landed. Populate it by
running the exporter against saved CRL runs you want to evaluate.
The first `--promote-default` invocation will set `detect-default` if
any bundle clears the floor.

## Current bundles

| Bundle | Frontend | pres_f1 | min_pres_f1 | source_run | Notes |
|--------|----------|--------:|------------:|------------|-------|
| _(empty — populate via the exporter)_ |  |  |  |  |  |
```

- [ ] **Step 3: Write `classify-bundles/README.md`**

Content:

```markdown
# Classify-side CRL Bundles

Pre-exported TorchScript bundles for `infer-classify`. Each bundle is a
self-contained directory:

```
<bundle-name>/
├── meta.json              # frontend_type, mode, sensors, sample rates,
│                          #   window sizes, z_dim, class_names, probe_mode,
│                          #   source_run, type_f1, min_type_f1
├── encoder_fused.ts       # fused mode: (x_audio, x_seismic) -> (z, pres_logit)
└── type_head_fused.ts     # (z) -> type_logits
```

Per-sensor mode bundles instead carry `encoder_audio.ts` +
`encoder_seismic.ts` + `type_head_audio.ts` + `type_head_seismic.ts`.

The classify pod re-encodes the inbound waveform — it does **not** rely
on the detect pod's encoder. That's why the encoder bytes are duplicated
in the classify bundle even when the detect bundle was built from the
same run.

## Selecting a bundle at build time

`scripts/build_containers.sh` reads `CLASSIFY_BUNDLE` to pick which
directory to copy into the `infer-classify` container. Default is
`classify-default` (a symlink, not committed yet — see Bootstrap below).

```bash
# Use the default
bash scripts/build_containers.sh infer-classify

# Pick a specific bundle
CLASSIFY_BUNDLE=multiscale-vae-<run>-mlp_ztype-v1 \
    bash scripts/build_containers.sh infer-classify

# Force a re-copy even if build/classify-export/ already has the same bundle
CRL_FORCE_EXPORT=1 bash scripts/build_containers.sh infer-classify
```

If `CLASSIFY_BUNDLE` doesn't resolve and `CLASSIFY_RUN_DIR` is set, the
build falls back to re-exporting from a saved CRL run.

## Naming convention

`<frontend>-<training-mode>-<run-id>-<probe>-v<N>`

Example: `multiscale-vae-2026_04_29_13_26_17-mlp_ztype-v1`

- frontend: `multiscale`, `morlet`, `morlet_per_sensor`, `morlet_fused`,
  `morlet_learnable`, `morlet_learnable_fused`
- training-mode: `vae`, `contrastive`, `disentangled`
- run-id: the saved-run directory name
- probe: `mlp_ztype`, `linear_ztype`, `linear_fullz`
- v<N>: monotonic version

## Default symlink

`classify-default` points at the current shipping leader. Promotion is
driven by `--promote-default` on `export_for_inference.py`.

## Selection rules

| Metric | Role | Value |
|--------|------|-------|
| `type_f1` | Primary | Highest on held-out eval (val) split |
| `min_type_f1` | Tie-breaker | Highest worst-location type F1 |
| `min_type_f1` | Promotion floor | `≥ 0.40` to be eligible for `classify-default` |

Tie band: `|type_f1_a − type_f1_b| < 0.01`. Inside the band the
tie-breaker fires; outside it the primary alone decides.

The capstone target is `type_f1 ≥ 0.70` and `min_type_f1 ≥ 0.50` —
the floor here protects against shipping a bundle that doesn't even
hit the cross-location minimum bar.

## Producing a new bundle

```bash
cd crl-train
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind classify \
    --bundle-name <frontend>-<mode>-<run>-<probe>-v1 \
    --promote-default
```

After exporting, commit the bundle directory and (if you promoted)
the symlink change. Update the catalog table below.

## Bootstrap

This catalog is empty after the restructure landed. Populate it by
running the exporter against saved CRL runs.

## Current bundles

| Bundle | Frontend | type_f1 | min_type_f1 | source_run | Notes |
|--------|----------|--------:|------------:|------------|-------|
| _(empty — populate via the exporter)_ |  |  |  |  |  |
```

- [ ] **Step 4: Delete the old `crl-bundles/` directory**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
rm -rf crl-bundles/
```

(This deletes the symlink + both bundle dirs + the README. Not destructive — the underlying TorchScript artifacts won't be regenerated from this directory; they get regenerated post-restructure by running the new exporter.)

- [ ] **Step 5: Verify**

```bash
ls inference-engine/ | grep -E 'bundles|crl-bundles'
```

Expected: `classify-bundles`, `detect-bundles`. No `crl-bundles`.

```bash
ls inference-engine/detect-bundles/
ls inference-engine/classify-bundles/
```

Expected: each contains `.gitkeep` and `README.md`.

- [ ] **Step 6: Commit**

```bash
git add inference-engine/detect-bundles/ inference-engine/classify-bundles/
git rm -r inference-engine/crl-bundles/
git commit -m "bundles: replace crl-bundles/ with empty detect/classify catalogs

Old single-catalog layout deleted. Two empty catalog dirs created with
their READMEs. The exporter (Task 10) is the only way to populate them;
no bundles are auto-migrated."
```

---

## Task 12: Update Helm chart values + chart README + top-level README

**Files:**
- Modify: `inference-engine/chart/values.yaml`
- Modify: `inference-engine/chart/README.md`
- Modify: `inference-engine/README.md`

- [ ] **Step 1: Add bundle-selection section to `chart/values.yaml`**

The chart values today have no bundle-selection block — bundles are picked at build time by env vars. We're documenting the build-time selectors here so customers know what knobs exist; nothing in the templates references these keys (they're documentation only, set on the build host's environment).

Insert this block after the `images:` block in `chart/values.yaml` (around line 33, before the `# ROS2 / DDS configuration` section header):

```yaml
# -----------------------------------------------------------------------------
# CRL bundle selection (build-time only)
#
# These are *not* runtime Helm values — Kubernetes never sees them. They
# document the build-host environment variables that scripts/build_containers.sh
# reads when staging the inference pods' CRL artifacts. The values here are
# for reference only; set them as shell env vars when you run the build.
#
# Each pod selects independently:
#   - infer-detect carries the encoder + presence head from DETECT_BUNDLE
#     (a directory under inference-engine/detect-bundles/).
#   - infer-classify carries encoder + type head from CLASSIFY_BUNDLE
#     (a directory under inference-engine/classify-bundles/).
#
# The two pods may use bundles from the same CRL run or different runs.
# Defaults are the *-default symlinks in each catalog (which point at the
# leaderboard winner per the selection rules in each catalog's README).
# -----------------------------------------------------------------------------
bundles:
  # Build-time: export DETECT_BUNDLE=<name> before running scripts/build_containers.sh.
  detect: detect-default
  # Build-time: export CLASSIFY_BUNDLE=<name> before running scripts/build_containers.sh.
  classify: classify-default
```

- [ ] **Step 2: Update `chart/README.md`**

Find the section about building images (around line 25-50 in the Quick start section). After the `for img in discovery ingestor ... do` loop, add a paragraph like:

```markdown
> **Per-pod CRL bundle selection.** `infer-detect` and `infer-classify`
> bake in their CRL bundle at build time. Each pod selects independently:
>
> ```bash
> # Defaults: detect-default and classify-default symlinks.
> bash scripts/build_containers.sh
>
> # Pin specific bundles:
> DETECT_BUNDLE=multiscale-vae-<run>-v1 \
> CLASSIFY_BUNDLE=multiscale-vae-<run>-mlp_ztype-v1 \
>     bash scripts/build_containers.sh infer-detect infer-classify
> ```
>
> Available bundles are listed in `inference-engine/detect-bundles/README.md`
> and `inference-engine/classify-bundles/README.md`.
```

(Insert at the natural place in the existing Quick start flow — typically right before "If your registry is private,".)

- [ ] **Step 3: Rewrite the "CRL inference deployment" section in the top-level `README.md`**

In `inference-engine/README.md`, find the "CRL inference deployment" section (around line 671). Replace the entire section through the end (lines 671-742) with:

```markdown
### CRL inference deployment

The pipeline deploys two independent CRL bundles — one per inference pod:

- **`infer-detect`** carries the (encoder + presence head) for vehicle
  detection. The bundle is selected at build time by `DETECT_BUNDLE`
  (default `detect-default`) and resolved against
  [`inference-engine/detect-bundles/`](detect-bundles/README.md).
- **`infer-classify`** carries the (encoder + type head) for vehicle
  classification. Selected by `CLASSIFY_BUNDLE` (default
  `classify-default`) against
  [`inference-engine/classify-bundles/`](classify-bundles/README.md).

The two pods are independent: each carries its own copy of the encoder
it pairs with, and the classify pod re-encodes the inbound waveform
rather than depending on the detect pod's latent.

Each inference image carries a self-contained TorchScript bundle —
a small directory of `.ts` files plus a `meta.json`. Customers don't
need `crl-train` installed; they just pick a bundle (or accept the
default `*-default` symlinks).

`scripts/build_containers.sh` reads `DETECT_BUNDLE` / `CLASSIFY_BUNDLE`
and resolves them against the per-pod catalogs.

Override the bundles for a customer-specific build:

```bash
DETECT_BUNDLE=<detect-bundle-name> CLASSIFY_BUNDLE=<classify-bundle-name> \
    bash scripts/build_containers.sh infer-detect infer-classify

# Roll the inference pods (Helm install)
kubectl rollout restart deploy/infer-detect deploy/infer-classify

# Or for raw-manifest install
kubectl rollout restart deploy/infer-detect deploy/infer-classify -n default
```

The capstone targets are `pres_f1 ≥ 0.85` and `type_f1 ≥ 0.70`. The
selection rules in each catalog README enforce floors below those
targets so a regression isn't auto-promoted into the shipping default.

The wire protocol carries no latent — `DetectionResult` has only
`sensor_data + vehicle_detected + confidence`. Classify re-encodes
from `sensor_data.acoustic_data` / `seismic_data` using its own
bundle's window sizes.

Per-sensor mode in `infer-detect` combines the two presence heads via
OR; in `infer-classify` the two type heads are averaged before argmax.

#### Producing a new bundle (dev only)

When a new training run beats the current leader, re-export and commit
the new bundle. This requires `crl-train` installed alongside
`inference-engine`. The exporter writes directly into the appropriate
per-kind catalog when given `--bundle-name`:

```bash
cd crl-train

# Detect bundle (encoder + presence head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind detect \
    --bundle-name <frontend>-<mode>-<run>-v<N>

# Classify bundle (encoder + type head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind classify \
    --bundle-name <frontend>-<mode>-<run>-<probe>-v<N>
```

To evaluate a bundle against the catalog and repoint its `*-default`
symlink if it wins, add `--promote-default`:

```bash
poetry run python export_for_inference.py \
    --save-dir <save-dir> \
    --bundle-kind classify \
    --bundle-name <name> \
    --promote-default
```

After exporting, commit the new bundle directory and update the
catalog table in the relevant README. See
[`crl-train/README.md`](../crl-train/README.md) for the full deployment
workflow.
```

- [ ] **Step 4: Update the validation-status footnote referring to CRL_RUN_DIR / CRL_BUNDLE in the README**

Find the footnote near the validation table (around line 23). Replace any mention of `CRL_RUN_DIR` / `CRL_BUNDLE` / `multiscale-default` with the new layout. Existing text:

```markdown
| Model accuracy | 🔄 Re-test needed | Earlier near-random predictions traced to an ingestor-side ADC scale mismatch with the CRL training contract (training uses raw ADC counts, ingestor was dividing by `2^(bits-1)`). Fixed at `src/ingestor/buffer.py` (default off; `ADC_SCALE_NORMALIZE=1` restores legacy). The default `CRL_RUN_DIR` in `scripts/build_containers.sh` now points at the leaderboard winner — `multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type` (pres_f1 0.875, type_f1 0.657, cross-location min_type_f1 0.436; capstone target is 0.70). Live re-evaluation pending after rebuilding images on the server. |
```

Replace with:

```markdown
| Model accuracy | 🔄 Re-test needed | Earlier near-random predictions traced to an ingestor-side ADC scale mismatch with the CRL training contract (training uses raw ADC counts, ingestor was dividing by `2^(bits-1)`). Fixed at `src/ingestor/buffer.py` (default off; `ADC_SCALE_NORMALIZE=1` restores legacy). After the bundle restructure landed (2026-05), the catalogs (`detect-bundles/`, `classify-bundles/`) are empty — populate them via `crl-train`'s exporter and run end-to-end against the new bundles to re-measure accuracy. |
```

- [ ] **Step 5: Smoke-check the docs**

```bash
grep -nE 'CRL_BUNDLE|CRL_RUN_DIR|crl-bundles/|multiscale-default' inference-engine/README.md inference-engine/chart/README.md
```

Expected: zero matches (all references migrated).

- [ ] **Step 6: Commit**

```bash
git add inference-engine/chart/values.yaml inference-engine/chart/README.md inference-engine/README.md
git commit -m "docs: document independent detect/classify bundle selection

values.yaml gains a 'bundles:' block describing the build-time selectors
(documentation only — Helm doesn't render these). chart README and
top-level README rewrite the CRL deployment section around DETECT_BUNDLE
and CLASSIFY_BUNDLE."
```

---

## Task 13: Acceptance gate — run the full test suite

**Why:** the spec's gate before declaring implementation done is `pytest tests/ -v` at baseline. Run it once with all changes in place to confirm.

- [ ] **Step 1: Run the test suite**

```bash
cd /Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/inference-engine
poetry run pytest tests/ -v
```

Expected: **167 passed, 3 skipped (rclpy-only), 0 failed.**

- [ ] **Step 2: If any test fails, fix the regression in this task before proceeding**

Common regressions to watch for:
- A `DetectionResult` test fixture still references `z_*` somewhere.
- A `meta.json` fixture missed the `class_names` / `probe_mode` cleanup.
- The classify pod's encoder mock returns the wrong shape — encoder returns `(z, pres_logit)`, not just `z`.

- [ ] **Step 3: Verify the pod source files don't reference dead names**

```bash
grep -nE 'z_fused|z_audio|z_seismic|crl-bundles|build/crl-export|CRL_BUNDLE\b|CRL_RUN_DIR\b|multiscale-default|update_default_symlink|update-default-symlink' \
    inference-engine/src/ inference-engine/scripts/ inference-engine/chart/ \
    crl-train/export_for_inference.py 2>/dev/null
```

Expected: zero matches in source/scripts/chart (the spec doc itself may still have historical references — that's fine since it's a snapshot of what got built).

- [ ] **Step 4: Commit nothing — this task is a gate, not a code change**

If steps 1-3 all pass, the implementation is complete. The next operator action (out of scope of this plan) is to run the exporter post-restructure to populate the catalogs.

---

## Task 14: Update the spec's status

**Files:**
- Modify: `docs/superpowers/specs/2026-05-04-inference-engine-independent-bundles-design.md`

- [ ] **Step 1: Update the status line**

Find the line near the top:

```markdown
**Status:** Design approved 2026-05-04. Ready for implementation plan.
```

Replace with:

```markdown
**Status:** Implemented 2026-05-04. Catalogs are empty; populate via the exporter.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-04-inference-engine-independent-bundles-design.md
git commit -m "spec: mark independent-bundles design as implemented"
```

---

## Out-of-scope follow-up (operator action, not part of this plan)

After this plan lands, the operator's next steps to make the new pipeline runnable end-to-end:

1. Run the exporter against each saved CRL run to populate `detect-bundles/` and `classify-bundles/`. Each run produces one detect bundle and one classify bundle via two invocations.
2. Run with `--promote-default` on the leader for each kind to set `detect-default` / `classify-default`.
3. Update each catalog's README table to reflect what's now in the catalog.
4. Run `scripts/local_smoke.sh` to confirm the pipeline starts cleanly with the new bundles.
5. Run `scripts/replay_in_kind.sh <audio> <seismic>` against a known recording to confirm end-to-end accuracy (replaces the validation-status table footnote's "🔄 Re-test needed" item).

---

## Self-review notes

After writing the plan, I reviewed it against the spec:

**Spec coverage:**
- §1 Goal & scope → covered by all tasks collectively.
- §2 On-disk bundle layout → Task 10 (exporter writes the layout), Task 11 (catalog skeletons + README documents the layout).
- §3.1 Proto change → Task 1.
- §3.2 infer_detect refactor → Task 4.
- §3.3 infer_classify refactor → Task 5.
- §3.3 cross-pod compatibility note → covered by the catalog READMEs (Task 11) which surface window sizes per bundle.
- §3.4 Dockerfiles → Tasks 7 + 8.
- §4.1 build_containers.sh + CRL_RUN_DIR split + hard-error → Task 9.
- §4.2 exporter --bundle-kind → Task 10.
- §4.3 --promote-default → Task 10 (steps 4, 7).
- §4.4 Helm values rename → Task 12 (step 1).
- §5.1 Test updates → Tasks 2, 3, 6.
- §5.2 Documentation updates → Tasks 11, 12.
- §5.3 No bundle migration → Task 11 (catalogs land empty).
- §5.4 Rollout sequence → matches Tasks 1-13.
- §6 Selection rules → Task 10 (steps 7) + Task 11 (READMEs document the rules).

**Placeholder scan:** No "TBD", no "implement later", no "similar to Task N" without code. Each step has either an exact command, an exact diff, or both.

**Type consistency:** `InferClassifyNode` constructor signature is `(nc, encoders, heads, meta, device)` in Task 5 (definition) and Task 6 (test usage). `_infer_fused` takes `(x_audio, x_seismic)` in both definition and test. `--bundle-kind` is the same flag name in Tasks 9 (build script call) and 10 (exporter flag definition). `_promote_default` is defined and called consistently. `<KIND>_BUNDLE_LABEL` build-arg name matches between Dockerfiles (Tasks 7, 8) and `build_containers.sh` (Task 9).
