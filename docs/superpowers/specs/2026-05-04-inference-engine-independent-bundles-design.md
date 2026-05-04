# Independent CRL Bundles for `infer-detect` and `infer-classify`

**Status:** Design approved 2026-05-04. Ready for implementation plan.

## 1. Goal & scope

Restructure `inference-engine/` so the two inference pods carry independent
(backbone + head) pairs:

- **`infer-detect`** carries the backbone + presence head from whichever
  CRL run wins on detection criteria.
- **`infer-classify`** carries the backbone + type head from whichever
  CRL run wins on classification criteria.

The two pods are selected by independent build-time env vars and may come
from the same CRL run or different ones. The classification pod becomes
self-sufficient: it re-encodes the raw waveform from the inbound message
rather than depending on the detection pod's latent.

**In scope:** the bundle catalog layout, the wire protocol between detect
and classify, both pod loaders, the build script, the `crl-train`
exporter's bundle-kind flag, the Helm chart's value keys, and the test
suite updates that follow from the proto change.

**Out of scope:** training/evaluation code, ingestor, discovery, egress
ROS2 wiring (egress is unaffected by the restructure beyond the deletion
of `z_*` fields from `DetectionResult`, which it does not read), and any
auto-generation of the catalog README tables (manual is fine for the
expected catalog sizes).

## 2. On-disk bundle layout

Replace the single `crl-bundles/` catalog with two parallel catalogs:

```
inference-engine/
├── detect-bundles/
│   ├── README.md                 # catalog table + producing-a-bundle docs
│   ├── detect-default            # symlink → leaderboard-best detect bundle
│   └── <name>/
│       ├── meta.json             # frontend_type, mode, sensors,
│       │                         #   audio_sample_rate, audio_window_size,
│       │                         #   seismic_sample_rate, seismic_window_size,
│       │                         #   z_dim, presence_threshold, source_run,
│       │                         #   pres_f1, min_pres_f1
│       └── encoder_fused.ts      # OR encoder_audio.ts + encoder_seismic.ts
├── classify-bundles/
│   ├── README.md                 # catalog table + producing-a-bundle docs
│   ├── classify-default          # symlink → leaderboard-best classify bundle
│   └── <name>/
│       ├── meta.json             # frontend_type, mode, sensors,
│       │                         #   audio_sample_rate, audio_window_size,
│       │                         #   seismic_sample_rate, seismic_window_size,
│       │                         #   z_dim, class_names, probe_mode,
│       │                         #   source_run, type_f1, min_type_f1
│       ├── encoder_fused.ts      # OR encoder_audio.ts + encoder_seismic.ts
│       └── type_head_fused.ts    # OR type_head_audio.ts + type_head_seismic.ts
```

Each bundle is self-contained. The classify bundle carries its own copy
of the encoder it pairs with even if those bytes match what's in some
detect bundle. There is no cross-bundle reference; each pod loads
exactly the directory it was given and never reads the other pod's
bundle. This is the structural property that makes independent
selection enforceable rather than aspirational.

**Bundle naming:**

- Detect: `<frontend>-<training-mode>-<run-id>-v<N>`
- Classify: `<frontend>-<training-mode>-<run-id>-<probe>-v<N>`

Detect names omit the probe component because the detect bundle has no
probe. The exporter validates the trailing `-v<N>` suffix on both
kinds.

**`crl-bundles/` is removed entirely.** The single `CRL_BUNDLE` env
var is retired; `build_containers.sh` errors out if it is set, with a
one-line message pointing at `DETECT_BUNDLE` / `CLASSIFY_BUNDLE`.

## 3. Wire protocol & pod refactor

### 3.1 Proto change

`inference-protos/protos/inference.proto`:

`DetectionResult` loses `z_fused`, `z_audio`, and `z_seismic`. After
the change it carries:

- `sensor_data` (the original window — already includes acoustic and
  seismic data that classify will re-encode)
- `vehicle_detected` (bool)
- `confidence` (float)

`compile_protos.py` regenerates `inference_pb2.py`; the regenerated
file is committed in the same change.

### 3.2 `infer_detect/main.py`

- `MODEL_DIR=/app/model` continues pointing at whatever bundle was
  COPY'd in. Loader logic is structurally unchanged: read `meta.json`,
  load `encoder_fused.ts` or both per-sensor encoders, run forward to
  produce `(z, pres_logit)`, sigmoid the logit, threshold against
  `presence_threshold`.
- Drops the `z` payload writes — no longer extends `result.z_fused` /
  `z_audio` / `z_seismic` (those fields no longer exist). The encoder
  still runs (the presence head needs the latent), but the latent is
  local to this pod.
- Published `DetectionResult` is `sensor_data + vehicle_detected +
  confidence`.

### 3.3 `infer_classify/main.py`

- Major change: the pod becomes a *full encode + classify* pipeline
  rather than a head-only pipeline.
- Loads both the encoder (`encoder_fused.ts` or per-sensor pair) AND
  the type head from its `MODEL_DIR`. The loader is structurally
  similar to `infer-detect`'s loader — same artifact-resolution code
  path can be shared if convenient, or duplicated if the symmetry is
  cleaner; both are acceptable.
- On a positive `DetectionResult`, the pod:
  1. Reads `detection.sensor_data.acoustic_data` /
     `detection.sensor_data.seismic_data` directly.
  2. Packs them into tensors using its *own* `meta.json`'s window
     sizes. The pod uses its own meta and ignores the detect pod's.
  3. Runs the encoder forward to produce `z`.
  4. Runs the type head, applies softmax, takes argmax.
- Negative `DetectionResult` messages are dropped, as today.
- Shape-mismatch errors (e.g. detect and classify bundles trained on
  different window sizes) surface as a clear log message; the pod
  drops the offending message rather than crashing.

**Cross-pod compatibility is the operator's responsibility.** The
selection rules in §6 do not enforce that the detect bundle and the
classify bundle were trained on compatible window sizes / sample
rates / sensors. If they aren't, classify will see shape mismatches
on every positive window and drop them. The catalog READMEs surface
the relevant fields so a mismatch is visible at promotion time
rather than runtime.

**Compute consequence:** classify now does an encoder forward pass per
positive window. On the cgroup-limited 2-CPU pod this is the same cost
as detect's encoder pass and within the existing latency budget. Worth
watching under load in smoke/replay output, but not a design gate.

### 3.4 Dockerfiles

**`src/infer_detect/Dockerfile`:**

- `COPY build/detect-export /app/model` (was `build/crl-export`).
- Add `LABEL com.darkcircle.detect_bundle=$DETECT_BUNDLE` via build
  arg, populated by `build_containers.sh`.

**`src/infer_classify/Dockerfile`:**

- `COPY build/classify-export /app/model`.
- Add `LABEL com.darkcircle.classify_bundle=$CLASSIFY_BUNDLE`.

The labels let `docker inspect <image>` show which bundle is baked in,
without needing to run the container.

## 4. Build script & exporter

### 4.1 `scripts/build_containers.sh`

Replace the single `stage_crl_export()` with a parameterized
`stage_bundle <kind>` taking `detect` or `classify`:

- `stage_bundle detect` → `build/detect-export/`, sourced from
  `DETECT_BUNDLE` (default `detect-default`) or, on miss, re-exported
  from `DETECT_RUN_DIR` via the `crl-train` exporter with
  `--bundle-kind detect`.
- `stage_bundle classify` → `build/classify-export/`, sourced from
  `CLASSIFY_BUNDLE` (default `classify-default`) or, on miss,
  re-exported from `CLASSIFY_RUN_DIR` via `--bundle-kind classify`.

The single `CRL_RUN_DIR` env var is split into `DETECT_RUN_DIR` and
`CLASSIFY_RUN_DIR`, mirroring the bundle selectors. If neither the
named bundle nor the `<KIND>_RUN_DIR` resolves to a valid path, the
script errors out with a message naming both attempted paths and the
two ways to fix it (set `<KIND>_BUNDLE` to a directory under
`<kind>-bundles/`, or install crl-train and point `<KIND>_RUN_DIR` at
a saved run) — same shape as the existing error messages in
`stage_crl_export()`, just per kind.

Each kind has its own stamp file (`build/detect-export/.bundle_id`,
`build/classify-export/.bundle_id`) so changing one selector re-stages
only that pod.

The per-image case statement decides which staging call to run:

```bash
case "$name" in
    infer-detect)   stage_bundle detect ;;
    infer-classify) stage_bundle classify ;;
esac
```

`CRL_TRAIN_PYTHON` env var is unchanged — same `crl-train` venv for
either kind's fallback path.

If `CRL_BUNDLE=...` is set in the environment, the script errors out
with a one-line message pointing at the new selectors. Hard error
rather than warning so silent fallthrough during a restructure is
caught.

### 4.2 `crl-train/export_for_inference.py`

Add `--bundle-kind {detect,classify}`. Required when `--out-dir` or
`--bundle-name` is passed.

- `--bundle-kind detect`: writes `meta.json` with the detect-pod
  fields (frontend / mode / sensors / sample rates / window sizes /
  z_dim / presence_threshold / source_run / pres_f1 / min_pres_f1)
  plus the encoder `.ts` file(s). Does *not* write the type head.
- `--bundle-kind classify`: writes `meta.json` with the classify-pod
  fields (the same backbone fields + class_names + probe_mode +
  source_run + type_f1 + min_type_f1) plus both the encoder `.ts`
  file(s) AND the type head `.ts` file(s).

When `--bundle-name` is passed, the kind also determines the
destination root: `--bundle-kind detect` writes into
`inference-engine/detect-bundles/<name>/`; `--bundle-kind classify`
into `inference-engine/classify-bundles/<name>/`. The trailing-`-v<N>`
name validation is unchanged.

A single CRL saved-run produces one detect bundle and one classify
bundle via two separate invocations. They are authored independently
and may end up under different `*-default` symlinks.

The metric fields are read from the saved run's `report.json` —
**not recomputed** from the TorchScript artifacts. If `report.json` is
missing the required field, the exporter errors out with a clear
message rather than writing a bundle without provenance.

### 4.3 `--promote-default` (replaces `--update-default-symlink`)

When invoked with `--promote-default`:

1. The exporter writes the new bundle (per `--bundle-kind`).
2. It walks the entire catalog for that kind
   (`detect-bundles/*` or `classify-bundles/*`), reading every
   bundle's `meta.json`.
3. It applies the selection rules in §6.
4. If the winner is different from the current `<kind>-default`
   target, it repoints the symlink and prints the change.
5. If no bundle clears the floor, it prints a clear
   `"no eligible bundle for <kind>-default (highest <metric>=<value>,
   floor=<floor>)"` message and exits non-zero on the promotion step.
   The just-written bundle still exists on disk and can be selected
   explicitly via `DETECT_BUNDLE=<name>` / `CLASSIFY_BUNDLE=<name>`.

The flag is opt-in — dev re-exports for ablations don't pass it and
never surprise-promote a worse bundle.

### 4.4 Helm chart

`chart/values.yaml` replaces today's single `crl.bundle` key with:

```yaml
bundles:
  detect: detect-default      # name under inference-engine/detect-bundles/
  classify: classify-default  # name under inference-engine/classify-bundles/
```

These are build-time selectors that customers pass to
`build_containers.sh`. They do not ship inside the chart — Helm doesn't
package or render the bundle bytes; they're baked into the images at
build time. Documented in `chart/README.md`.

## 5. Tests, docs, and rollout

### 5.1 Test updates (in the same change)

- **`tests/test_infer_classify.py`:** rebuild fixtures so the classify
  pod loads encoder + type head from a fake `MODEL_DIR`, and
  `DetectionResult` instances populate
  `sensor_data.acoustic_data` / `sensor_data.seismic_data` instead of
  `z_*`. Mocked TorchScript modules gain an encoder forward pass.
- **`tests/test_infer_detect.py`:** drop assertions about `z_*` being
  written into the published `DetectionResult`. The detect pod still
  runs its encoder, but the latent doesn't leave the pod. Update
  meta.json fixtures to drop `class_names` / `probe_mode` (and to
  carry the new metric fields if any tests assert on them).
- **`tests/test_egress.py`:** any `DetectionResult` constructed in
  tests drops `z_*` fields (they no longer exist in the regenerated
  proto).

The acceptance gate before declaring implementation done is
`pytest tests/ -v` at the current baseline:
**167 passed, 3 skipped (rclpy-only), 0 failed.**

### 5.2 Documentation updates

- **`inference-engine/README.md`:** rewrite the "CRL inference
  deployment" section, the bundle catalog references, the
  `CRL_BUNDLE` mentions in `scripts/build_containers.sh`, and the
  validation-status table footnote.
- **`chart/README.md`:** update the bundle-selection section to
  describe the two independent selectors.
- **`crl-bundles/README.md`** is **deleted**. Replaced by:
  - `detect-bundles/README.md` — catalog table (frontend, `pres_f1`,
    `min_pres_f1`, source run pointer, notes column) plus
    producing-a-bundle docs.
  - `classify-bundles/README.md` — same structure with `type_f1`,
    `min_type_f1`.
- The "Producing a new bundle (dev only)" section in each new README
  documents the `--bundle-kind` and `--promote-default` flags.

### 5.3 No bundle migration

The two existing artifacts under `crl-bundles/` are **not migrated**.
They are deleted in the same commit that introduces the restructure.

After the restructure lands, `detect-bundles/` and `classify-bundles/`
exist as empty catalogs (each containing `README.md` and a `.gitkeep`
so git tracks the empty directory). No `*-default` symlinks exist
yet.

The first bundles are produced post-restructure by running the new
exporter against the saved CRL runs the operator wants to evaluate.
Each run produces one detect bundle and one classify bundle via two
exporter invocations. After enough bundles are in each catalog,
`--promote-default` re-evaluates and points the `*-default` symlinks
at the leaderboard winners per pod.

### 5.4 Rollout sequence within the implementation plan

1. Proto edit → recompile → commit regenerated `inference_pb2.py`.
2. Test updates that follow from the proto change (suite stays green).
3. Pod refactors: detect drops `z_*` writes; classify gains encoder
   load + forward pass.
4. Build script split (`stage_bundle <kind>` × 2) + Dockerfile `COPY`
   paths + image labels + Helm values rename to
   `bundles.detect` / `bundles.classify`. Hard error on `CRL_BUNDLE`.
5. Exporter `--bundle-kind` flag + `--promote-default` rewrite,
   including the `report.json`-reading logic for metric provenance.
6. Delete `crl-bundles/` + create `detect-bundles/` and
   `classify-bundles/` skeletons (READMEs + `.gitkeep`).
7. README rewrites (`inference-engine/README.md`, `chart/README.md`,
   the two new catalog READMEs).
8. **Acceptance gate (still part of the implementation deliverable):**
   `pytest tests/ -v` at baseline — **167 passed, 3 skipped (rclpy-only),
   0 failed.** No bundles need to exist in the new catalogs at this
   point; the test suite uses fixtures, not real bundles.

**Out of the implementation plan, but the next operator action:**
populate the new catalogs by running the exporter against saved CRL
runs, then validate end-to-end with `scripts/local_smoke.sh` and
`scripts/replay_in_kind.sh` against a known recording. This is
operator work that depends on the implementation landing, not part of
the implementation work itself.

## 6. Bundle selection & default-symlink promotion

### 6.1 Per-bundle metric provenance

Metrics live in each bundle's `meta.json`, written by the exporter
from the saved run's `report.json`:

- **Detect bundle:** `pres_f1`, `min_pres_f1`, `source_run`.
- **Classify bundle:** `type_f1`, `min_type_f1`, `source_run`.

These are records of what the source CRL run scored on its held-out
eval (val) split — same numbers `crl-train`'s eval pipeline produced.
Not recomputed at export time. Missing required fields in
`report.json` cause the exporter to error out.

### 6.2 Selection rules

| Pod | Primary metric | Tie band | Tie-breaker | Default-promotion floor |
|-----|---------------|----------|-------------|-----|
| detect | `pres_f1` (eval/val) | within ε = 0.01 | `min_pres_f1` (cross-location) | `pres_f1 ≥ 0.80` |
| classify | `type_f1` (eval/val) | within ε = 0.01 | `min_type_f1` (cross-location) | `min_type_f1 ≥ 0.40` |

Two runs are tied iff `|primary_a − primary_b| < 0.01`. The
tie-breaker only fires inside that band. If runs remain tied even on
the tie-breaker (numerically essentially impossible at float
precision), the alphabetically first bundle name wins — deterministic,
not meaningful.

The detect tie-breaker uses `min_pres_f1` (worst-location presence
F1), so the detect bundle has no dependency on the type head's
quality. The detect bundle's `meta.json` carries only presence-side
metrics.

### 6.3 Eligibility for `*-default`

The selection floors gate **only the default-symlink promotion**, not
bundle existence. A run scoring below the floor still produces a
bundle that lives in the catalog; it just is not auto-promoted to
`<kind>-default`. Operators can still pick it explicitly with
`DETECT_BUNDLE=<name>` or `CLASSIFY_BUNDLE=<name>` if needed.

If no bundle in the catalog clears the floor when `--promote-default`
runs, the symlink is left untouched (or absent, on a fresh catalog),
and the exporter exits non-zero on the promotion step with a clear
message naming the highest-observed metric value and the floor.

### 6.4 Bootstrap

The first bundles are produced after the restructure lands. Example
invocations:

```bash
cd crl-train

# Produce a detect bundle from a saved run, promote if it wins.
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind detect \
    --bundle-name <frontend>-<mode>-<run>-v1 \
    --promote-default

# Produce a classify bundle from the same (or a different) run.
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind classify \
    --bundle-name <frontend>-<mode>-<run>-<probe>-v1 \
    --promote-default
```

Each pod's `*-default` symlink is repointed independently, so the
detect winner and the classify winner may come from different runs.

### 6.5 Catalog README tables

`detect-bundles/README.md` lists every bundle with `pres_f1`,
`min_pres_f1`, `source_run`, and a "Notes" column.
`classify-bundles/README.md` lists `type_f1`, `min_type_f1`,
`source_run`, "Notes."

Auto-generation from `meta.json` is **not** in scope for this work —
the tables are updated by hand at promotion time. Auto-generation is
a candidate follow-up if and when the catalog grows large enough that
manual upkeep becomes friction.
