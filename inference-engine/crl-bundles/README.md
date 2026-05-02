# CRL Inference Bundles

Pre-exported TorchScript artifacts ready to bake into the inference
images. Each bundle is a self-contained directory:

```
<bundle-name>/
├── meta.json              # frontend_type, mode, sample rates, class names
├── encoder_fused.ts       # fused mode: (x_audio, x_seismic) → (z, pres_logit)
└── type_head_fused.ts     # (z) → type_logits
```

Per-sensor mode bundles instead carry `encoder_audio.ts`,
`encoder_seismic.ts`, `type_head_audio.ts`, `type_head_seismic.ts`.

## Selecting a bundle at build time

`scripts/build_containers.sh` reads `CRL_BUNDLE` to pick which directory
to copy into the container build context. Default is
`multiscale-default`.

```bash
# Use the default
bash scripts/build_containers.sh

# Pick a specific bundle
CRL_BUNDLE=multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v1 \
    bash scripts/build_containers.sh

# Force a re-copy even if build/crl-export/ already has the same bundle
CRL_FORCE_EXPORT=1 bash scripts/build_containers.sh
```

If `CRL_BUNDLE` doesn't resolve (the directory doesn't exist), the
build falls back to re-exporting from `CRL_RUN_DIR` if `crl-train`'s
Python venv is available — that's the dev path. Customers without
`crl-train` should always use a pre-bundled artifact.

## Naming convention

`<frontend>-<training-mode>-<run-id>-<probe>-<aux-flag>-v<N>`

Example: `multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v1`

- frontend: `multiscale`, `morlet`, `morlet_per_sensor`, `morlet_fused`,
  `morlet_learnable`, `morlet_learnable_fused`
- training-mode: `vae`, `contrastive`, `disentangled`
- run-id: the saved-run directory name
- probe: `mlp_ztype`, `linear_ztype`, `linear_fullz`
- aux-flag: `aux_type` if the probe was trained with the auxiliary
  type head, omit otherwise
- v<N>: monotonic version, bump when re-exporting from a different
  checkpoint of the same configuration

## Aliases

`multiscale-default` is a symlink to whichever bundle is the current
shipping default. Update it after promoting a new bundle. Customers
who don't override `CRL_BUNDLE` get whatever this points at.

## Producing a new bundle (dev only)

Customers don't run this — it requires `crl-train` and the saved-run
directory it points at. The export script writes directly into this
directory when given `--bundle-name`:

```bash
cd crl-train
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-name <frontend>-<mode>-<run>-<probe>-aux_type-v<N>
```

`--bundle-name` validates that the name ends in `-v<N>` and resolves
to `../inference-engine/crl-bundles/<name>/`. The script prints a
suggested `ln -sfn ...` line if the bundle should become the new
shipping default; or pass `--update-default-symlink` to repoint
`multiscale-default` automatically:

```bash
cd crl-train
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-name <frontend>-<mode>-<run>-<probe>-aux_type-v<N> \
    --update-default-symlink
```

After exporting, commit the bundle directory and (if you promoted)
the symlink change. Also update the catalog table below.

For one-off exports outside the bundle catalog (ad-hoc evaluation,
parity testing, etc.), use the original `--out-dir <path>` flag —
it skips the naming convention check and writes wherever you ask.

## Current bundles

`pres_f1` / `type_f1` are val-split numbers from the run's `report.json`;
`min_type_f1` is the worst per-class F1 on the full eval split.

| Bundle | Frontend | pres_f1 | type_f1 | min_type_f1 | Notes |
|--------|----------|--------:|--------:|------------:|-------|
| `multiscale-vae-2026_04_29_13_26_17-mlp_ztype-aux_type-v1` | multiscale | 0.873 | 0.664 | 0.433 | Current default. Backed by the 2026-04-29 13:26 run; `d_z=32`, `D_TYPE=12` (export rebinds the live latent constants from the checkpoint). |
| `multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v1` | multiscale | 0.875 | 0.657 | 0.436 | Prior default. Backed by the v3_lowfreq run, `d_z=24`, `D_TYPE=6`. |

`multiscale-default` → `multiscale-vae-2026_04_29_13_26_17-mlp_ztype-aux_type-v1`
