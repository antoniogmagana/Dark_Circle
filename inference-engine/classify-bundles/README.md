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
