# CRL-Train

Causal Representation Learning for multi-modal vehicle detection. A modular VAE/contrastive framework with a pluggable frontend, pluggable prior, and pluggable training mode. CITRIS-style temporal intervention matching replaces earlier NOTEARS-style causal discovery.

---

## What this repo does

Trains a latent representation from paired audio (16 kHz) and seismic (200 Hz) sensor windows. The representation has a **causal latent structure**: dedicated blocks for presence, vehicle type, proximity, and environment, plus a free noise block. Downstream probe heads (linear, MLP, or full-latent) are trained on the frozen representation to predict presence + vehicle type.

```
raw audio  ┐                                                ┌→ presence
           ├→ Morlet / Multiscale frontend → Transformer ───┤→ vehicle type (4 classes)
raw seismic┘          (fused or per-sensor)   encoder       ├→ proximity
                                              (VAE or       └→ (free nuisance)
                                               contrastive)
```

The model is **not hardcoded to a specific frontend or training objective**. Six frontends and two training modes are registered; any valid combination runs through the same `Trainer`.

---

## Architecture (current, as of 2026-04-24)

### Six frontend variants

| Frontend | Fusion | Kernels | Per-sensor params | Notes |
|---|---|---|---|---|
| `multiscale` | Early | Learnable Conv1D, 3 kernel sizes | Shared `d_model` | Fastest, strongest aggregate F1 today |
| `morlet` | Late | Fixed analytic Morlet wavelets | SR-derived freq range | Legacy; defaults to broad-band heuristic |
| `morlet_per_sensor` | Late | Fixed Morlet, per-sensor freq bands | `morlet_per_sensor_params` | Coupled kernel/stride derivation |
| `morlet_fused` | Early | Fixed Morlet, per-sensor freq bands | Same as above | Morlet kernels + multiscale topology |
| `morlet_learnable` | Late | Learnable log-space scales | Same as above | Stage-2 init optional |
| `morlet_learnable_fused` | Early | Learnable scales, early fusion | Same as above | Combines learnable + fusion |

All Morlet variants use **FFT-based convolution** when kernel_size ≥ 512 (automatic; ~3× faster on CPU for audio kernel size 4585).

### Two training modes

- **`vae`** — classical CRL. ELBO + adaptive β annealing + intervention matching + aux heads. Prior is pluggable: `standard` (N(0,I)) or `conditional` (iVAE-style label-conditioned MLP for identifiability).
- **`contrastive`** — NT-Xent over `StratifiedPairDataset` partners. Same-type and consecutive-window partners are positives; different-type and cross-dataset are negatives. No decoder / KL / aux during CRL; downstream probes train post-hoc on the frozen encoder.

### Two-stage training

`--init-from-run PATH|auto` loads a converged fixed-Morlet checkpoint (`morlet_per_sensor` or `morlet_fused`), upgrades it to the learnable variant, and fine-tunes. Encoder runs at `lr × stage2_encoder_lr_mult` (default 0.3×) while filters are the primary mover at `lr × morlet_learnable_lr_mult` (default 0.1×). 3-epoch linear warmup on filter LR to avoid shocking the trained encoder.

Rationale: learnable scales trained from scratch chase a moving target (the encoder is drifting too). Two-stage separates "learn a representation" from "tune the filterbank" so the filters see a stable target.

### Sweep tooling

`run_experiments.py --sweep configs/sweeps/<file>.yaml` runs a YAML-driven sweep. Each run spawns `train.py` as a subprocess (GPU + crash isolation); results aggregate into `summary.csv` + `summary.json`. See `configs/sweeps/frontend_comparison.yaml` for the reference sweep over all 6 frontends + phase + learnable_w0.

### Analysis scripts

Post-hoc comparison of completed runs:

- `compare_runs.py` — leaderboard CSV + markdown
- `compare_ablations.py` — pairwise deltas along predefined axes
- `compare_cross_location.py` — per-dataset heatmap + ship metric (min-F1)
- `plot_run.py` — per-run diagnostic plots (training curves, beta, Morlet freq drift, downstream)
- `plot_aggregate.py` — cross-run overlay + bar + scatter

All read recursively from `saved_crl/runs/` and write to `saved_crl/analysis/` by default.

---

## Directory layout

```
crl-train/
├── crl_vehicle/                 # core library (see crl_vehicle/README.md)
│   ├── config.py                # CRLConfig — single source of truth
│   ├── analysis.py              # post-hoc run readers
│   ├── stage2.py                # two-stage helpers: find_compatible_run + conversion
│   ├── data/                    # SensorDataset + StratifiedPairDataset
│   ├── losses/                  # reconstruction, KL, intervention, NT-Xent
│   ├── models/                  # frontends, encoder/decoder, latent split, heads
│   ├── priors/                  # Prior ABC, Standard, Conditional
│   ├── probe/                   # log-prior recalibration
│   └── training_modes/          # TrainingMode ABC, VAE, Contrastive, factory
├── training/
│   └── trainer.py               # CRLModel + Trainer (epoch loop, opt groups, stage-2)
├── configs/
│   └── sweeps/                  # YAML sweep specs
├── tests/                       # 354 tests mirror the module tree
├── saved_crl/                   # run outputs (see analysis scripts)
│   ├── runs/                    # canonical: <frontend>/<training_mode>/<run-id>/{crl,downstream,eval}/
│   │   ├── multiscale/
│   │   ├── morlet_per_sensor/
│   │   ├── morlet_fused/
│   │   ├── morlet_learnable/
│   │   ├── morlet/
│   │   ├── supervised/          # non-CRL baselines: id_split/, file_split/
│   │   └── _archive/            # smoke tests, sweeps, old-layout dirs
│   ├── caches/
│   │   ├── waveform/            # SensorDataset feature cache
│   │   └── id_split/            # ID-split manifest cache
│   ├── analysis/                # compare_runs.py / plot_aggregate.py outputs
│   └── slides-figs/             # session figures, performance tables, notes
│
├── train.py                     # single-run CLI
├── run_full_diagnostic.py       # CRL + downstream + cross-dataset eval in one shot
├── run_experiments.py           # hardcoded list + YAML-driven sweeps
├── eval.py                      # standalone eval of a CRL run on a dataset split
├── validate_run.py              # smoke validation of a completed run
├── smoke_test.py                # tiny synthetic-batch sanity check
│
├── compare_runs.py              # leaderboard
├── compare_ablations.py         # axis deltas
├── compare_cross_location.py    # per-dataset heatmap
├── plot_run.py                  # per-run plots
├── plot_aggregate.py            # cross-run plots
```

---

## Quick start

```bash
# Full pipeline: CRL + downstream probes + cross-dataset eval.
python run_full_diagnostic.py --frontend multiscale --crl-epochs 100

# Single frontend, VAE + standard prior, 100 epochs.
python train.py --frontend morlet_per_sensor --crl-epochs 100 \
    --config-overrides-json '{"morlet_use_phase": true}'

# Contrastive NT-Xent, multiscale backbone.
python train.py --frontend multiscale --training-mode contrastive --crl-epochs 100

# Two-stage learnable Morlet (auto-finds the most recent compatible fixed-Morlet run).
python train.py --frontend morlet_learnable \
    --init-from-run auto --crl-epochs 50 \
    --config-overrides-json '{"morlet_use_phase": true}'

# YAML-driven sweep (subprocess per run; GPU/crash isolation).
python run_experiments.py --sweep configs/sweeps/frontend_comparison.yaml

# Post-hoc analysis (run any time after runs finish).
python compare_runs.py
python compare_cross_location.py
python plot_aggregate.py
```

---

## Configuration

`crl_vehicle/config.py::CRLConfig` is the single source of truth. Every knob exposed to the model lives here as a dataclass field; CLI flags are thin wrappers. Major fields:

| Field | Default | Role |
|---|---|---|
| `d_z` | 24 | Latent dim; must exceed `D_CAUSAL = 19` to leave a free subspace |
| `d_model` | 64 | Transformer hidden size + frontend output channels |
| `n_layers` / `n_heads` | 2 / 4 | Transformer shape |
| `frontend_type` | `"multiscale"` | See frontend variants above |
| `training_mode` | `"vae"` | `"vae"` or `"contrastive"` |
| `prior_type` | `"standard"` | `"standard"` or `"conditional"` (iVAE) |
| `morlet_use_phase` | `False` | Emit `[log_power, cos_phase, sin_phase]` → 3× channels |
| `morlet_learnable_w0` | `False` | Learnable per-filter bandwidth (only for `morlet_learnable*`) |
| `morlet_learnable_lr_mult` | 0.1 | Filter LR = `lr × this` |
| `stage2_encoder_lr_mult` | 0.3 | Encoder LR when loading a stage-1 checkpoint |
| `n_epochs` | 100 | CRL training epochs |
| `batch_size`, `lr`, `wd` | 128, 3e-4, 1e-4 | Standard optimization hyperparams |
| `early_stop_patience` | 25 | Epochs without val improvement before stopping |
| `kl_floor`, `kl_target`, `beta_step` | 0.01, 0.5, 0.02 | Adaptive-β schedule |
| `use_focal_type` | `False` | When `True`, replace type CE with `(1 - p_t)^γ · weighted_CE` in pretraining aux_type and downstream probe. Stacks on existing `type_class_weights`; presence BCE unaffected |
| `focal_type_gamma` | 2.0 | Focal exponent γ. Ignored unless `use_focal_type=True` |

Per-sensor Morlet knobs live in `morlet_per_sensor_params`:

```python
{
    "audio":   {"freq_min": 20.0,  "freq_max": 8000.0, "w0": 6.0,
                "out_channels_frac": 1.0, "target_tokens": 32,
                "receptive_cycles": 3.0},
    "seismic": {"freq_min": 2.0,   "freq_max": 40.0,   "w0": 6.0,
                "out_channels_frac": 1.0, "target_tokens": 32,
                "receptive_cycles": 3.0},
}
```

`kernel_size` and `pool_stride` are **derived** from these — not hardcoded. See `crl_vehicle/models/README.md` for the formulas.

---

## Run outputs

Each run writes to a `save_dir` (default: `saved_crl/runs/<frontend>/<training_mode>/<timestamp>/`):

```
<save_dir>/
├── crl/
│   ├── meta.json                    # config + sensors + stage2 attribution
│   ├── crl_metrics.csv              # per-epoch training curves
│   ├── crl_checkpoint_summary.json  # best ELBO, best aux_type_f1, learned_morlet_params
│   ├── crl_best.pth                 # val_ref_elbo-selected checkpoint
│   ├── crl_best_aux_type.pth        # val_aux_type_f1-selected checkpoint
│   ├── crl_final.pth                # last-epoch checkpoint
│   └── learnable_morlet_freqs.csv   # (only for learnable variants)
├── downstream/<probe_mode>__<ckpt>/
│   ├── downstream_metrics.csv       # per-epoch probe training
│   ├── downstream_best.pth
│   └── meta.json
└── eval/<probe_mode>__<ckpt>/<split>/
    ├── eval_report.json             # presence + type metrics
    └── confusion_*.png
```

Splits: `iobt`, `focal`, `m3nvc` (per-dataset), `full` (pooled).
Probe modes: `linear_ztype`, `mlp_ztype`, `linear_fullz`.
Checkpoint names: `crl_best.pth` (ELBO), `crl_best_aux_type.pth` (F1 — epoch-invariant, preferred when training uses β annealing).

---

## Deploying a checkpoint to the inference engine

Use `export_for_inference.py` to convert a probe-trained run into the
TorchScript bundle the inference pods consume. The exporter writes
**directly into `inference-engine/crl-bundles/`** when given
`--bundle-name`, so the deployment artifact lands where
`scripts/build_containers.sh` will find it via `CRL_BUNDLE`. Assumes
`crl-train` and `inference-engine` are siblings under one parent
directory (the standard project layout).

```bash
# Pick the run + probe, choose a versioned bundle name, write into the
# inference-engine catalog.
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-name <frontend>-<mode>-<run>-<probe>-aux_type-v<N>
```

Naming convention (enforced by the exporter — fails fast if the name
doesn't end in `-v<N>`): `<frontend>-<training_mode>-<run-id>-<probe>-[aux_type-]v<N>`.

Example for the current shipping leader:

```bash
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type \
    --bundle-name multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v2
```

To promote the new bundle as the default that customers get when
they don't override `CRL_BUNDLE`, repoint the symlink in the same
invocation:

```bash
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type \
    --bundle-name multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v2 \
    --update-default-symlink
```

After exporting:

1. Update the catalog table in `inference-engine/crl-bundles/README.md`
   with the new bundle's pres_f1 / type_f1 / min_type_f1 numbers.
2. Commit the bundle directory and (if you promoted) the symlink change.

For ad-hoc exports outside the bundle catalog (parity testing, scratch
deploys), use `--out-dir <path>` instead of `--bundle-name` — that
flag skips the naming check and writes wherever you point it.

---

## Testing

```bash
python -m pytest --ignore=smoke_test.py -q
```

354 tests across `tests/` mirroring the module tree. Skip `smoke_test.py` because it's a script, not a pytest module (historical name collision).

---

## Where to read next

Start with the nested READMEs for specifics on each layer:

- **`crl_vehicle/README.md`** — package overview, config dataclass, post-hoc analysis helpers
- **`crl_vehicle/data/README.md`** — dataset format, stratified partner sampling, interventions
- **`crl_vehicle/models/README.md`** — all 6 frontends, encoder/decoder, latent split, heads
- **`crl_vehicle/losses/README.md`** — reconstruction, KL, intervention matching, NT-Xent
- **`crl_vehicle/priors/README.md`** — Prior ABC, StandardPrior, ConditionalPrior (iVAE)
- **`crl_vehicle/probe/README.md`** — log-prior recalibration (target-prior shift)
- **`crl_vehicle/training_modes/README.md`** — TrainingMode ABC, VAE, Contrastive, factory
- **`training/README.md`** — Trainer class, CRLModel assembly, two-stage mechanics
