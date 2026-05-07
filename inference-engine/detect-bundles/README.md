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
| `pres_mcc` | Primary | Highest test-time presence MCC on the `full` split |
| `min_pres_f1` | Tie-breaker | Highest worst-location presence F1 |
| `pres_mcc` | Promotion floor | `≥ 0.40` to be eligible for `detect-default` |

Tie band: `|pres_mcc_a − pres_mcc_b| < 0.01`. Inside the band the
tie-breaker fires; outside it the primary alone decides.

**Why MCC, not raw F1**: the test set is ~75% positive (vehicles present).
Raw F1 over a positive-skewed split rewards a model that just predicts
"yes" — recall stays high, F1 looks great, specificity collapses. MCC is
invariant to the class prior, so it ranks calibration honestly. The
2026-05-03 cycle exposed this concretely: `2026-05-03_15-26-22` had the
highest F1 (0.879) but specificity = 0.43 (the model flagged 57% of empty
windows as vehicles); `2026-05-03_05-02-44` had lower F1 (0.858) but
specificity = 0.65, balanced_accuracy = 0.75, and **higher MCC (0.475 vs
0.439)** — i.e. genuinely better detection.

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

## Current bundles

| Bundle | Frontend | pres_MCC | min_pres_f1 | bal_acc | pres_F1 | source_run | Notes |
|--------|----------|---------:|------------:|--------:|--------:|------------|-------|
| `multiscale-vae-2026_05_03_05_02_44-v1` | multiscale | **0.475** | 0.617 | **0.746** | 0.858 | `2026-05-03_05-02-44` | d_z=24. Current `detect-default`. Wins on pres_MCC under the MCC-primary selection rule. |
| `multiscale-vae-2026_05_03_15_26_22-v1` | multiscale | 0.439 | **0.822** | 0.684 | **0.879** | `2026-05-03_15-26-22` | d_z=32. Previous `detect-default` (promoted under the legacy `pres_f1`-primary rule). Highest raw F1 and best location-robustness, but recall-biased: specificity collapses to 0.43 on the imbalanced full split. |

`detect-default` → `multiscale-vae-2026_05_03_05_02_44-v1`
