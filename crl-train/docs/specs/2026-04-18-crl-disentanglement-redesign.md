# CRL Disentanglement Redesign

**Date:** 2026-04-18
**Status:** Approved

## Context

The existing CRL pipeline (CITRIS-style VAE with multiscale/morlet frontends) fails to disentangle vehicle signal from environmental signal. Downstream linear probes trained on the frozen backbone produce chance-level predictions across all 26 evaluated epochs. Root causes identified:

1. Intervention signal is noise augmentation type — classifier learns "which noise was added" not "what causal variable changed between timesteps"
2. `CausalLatentSpace.split()` applies `sigmoid`/`softmax` before probe inputs, saturating gradients and capping probe capacity
3. No auxiliary supervision during pre-training — latent slot assignments are arbitrary, nothing pins dims to semantic roles
4. `d_pres=1` gives presence a single scalar with minimal discriminative capacity
5. `run_experiments.py` never calls downstream, has a missing `csv` import (line 179 crash), and passes pair loaders to downstream
6. Hardware config is hardcoded for H100 with no graceful degradation

This redesign implements Option B: supervised auxiliary losses + intervention signal redesign + latent space expansion, keeping the VAE/ELBO structure intact.

---

## 1. Intervention Signal Redesign

### Current
`interv_idx ∈ {0..N}` selects a noise augmentation applied to both t and t+1. `UnknownInterventionClassifier` learns to predict augmentation type — structurally unrelated to causal factorization.

### Redesign
Replace intervention target with a **2-bit label-change vector** derived from ground-truth labels:

- `pres_changed`: `detection_label_t != detection_label_tn` (bool)
- `type_changed`: `vehicle_type_t != vehicle_type_tn` (bool)

`UnknownInterventionClassifier` is redesigned to predict this 2D binary vector from `(z_t, z_tn)` using **binary cross-entropy per bit** (not softmax). This is structurally valid CITRIS pressure: the classifier learns "which causal variable changed," pushing presence into one subspace and type into another.

Noise augmentations are **retained as data augmentation only** — applied independently at t and t+1, decoupled from the intervention target. This maintains environmental generalization without confusing causal structure.

### Dataset Changes
`ConsecutivePairDataset.__getitem__` must emit:
- `detection_label_t`, `detection_label_tn`
- `vehicle_type_t`, `vehicle_type_tn`

`collate_pairs` must include these four new keys as `torch.long` tensors.

---

## 2. Latent Space Redesign

### Dimensions
`d_z = 24`, all slots are **raw slices** — no nonlinearities in `split()`.

| Slot | Dims | Size | Role | Pre-training supervision |
|---|---|---|---|---|
| `z_pres` | 0–3 | 4 | Vehicle presence | Auxiliary BCE |
| `z_type` | 4–9 | 6 | Vehicle type/class | Auxiliary CE |
| `z_prox` | 10–12 | 3 | Proximity/amplitude | RMS amplitude MSE |
| `z_env` | 13–18 | 6 | Environmental/noise factors | Intervention change signal |
| `z_free` | 19–23 | 4 | Unconstrained remainder | None |

### `CausalLatentSpace.split()` returns raw slices
```python
z_pres = z[..., 0:4]
z_type = z[..., 4:10]
z_prox = z[..., 10:13]
z_env  = z[..., 13:19]
z_free = z[..., 19:24]
```
Downstream heads and auxiliary losses apply their own nonlinearities internally.

### Head dimensions
- `LinearPresenceHead`: `Linear(4, 1)`
- `LinearTypeHead`: `Linear(6, n_classes)`
- `LinearProximityHead`: `Linear(3, 1)` — added for future range-label supervision

### `UnknownInterventionClassifier`
Input: `cat(z_env_t, z_env_tn)` → `Linear(12, 2)`. Predicts `[pres_changed, type_changed]` with BCE loss per bit.

---

## 3. Auxiliary Supervision During Pre-Training

Three lightweight auxiliary heads attached to `CRLModel`, used only during CRL pre-training, discarded before downstream training.

| Loss | Head | Target | Weight | Mask |
|---|---|---|---|---|
| `aux_pres` | `Linear(4, 1)` on `z_pres` | `detection_label` (BCE) | `lambda_aux_pres = 0.3` | all samples |
| `aux_type` | `Linear(6, n_classes)` on `z_type` | `vehicle_type` (CE) | `lambda_aux_type = 0.3` | `detection_label==1` and `vehicle_type>=0` |
| `aux_prox` | `z_prox.mean(dim=-1)` scalar | batch-normalized RMS amplitude (MSE) | `lambda_aux_prox = 0.1` | all samples |

**RMS amplitude computation:** compute `rms = x.pow(2).mean(dim=-1).sqrt()` on the raw waveform before the frontend, then batch-normalize to `[0,1]` using `(rms - rms.min()) / (rms.max() - rms.min() + 1e-8)`.

**Total pre-training loss:**
```
L = recon + beta*kl + lambda_interv*interv
  + lambda_aux_pres*aux_pres
  + lambda_aux_type*aux_type
  + lambda_aux_prox*aux_prox
```

Auxiliary heads are initialized fresh each run and are not saved as checkpoints (they're scaffolding, not outputs).

---

## 4. Experiment Grid (`run_experiments.py`)

Five experiments ablating the changes that matter for disentanglement:

| Name | Frontend | `lambda_interv` | Aux supervision | Intervention signal | Purpose |
|---|---|---|---|---|---|
| `exp1_baseline` | multiscale | 1.0 | off | noise type (current) | Current behavior baseline |
| `exp2_aux_on` | multiscale | 1.0 | on | noise type | Aux supervision contribution in isolation |
| `exp3_redesigned` | multiscale | 1.0 | on | label-change | Full redesign, default weight |
| `exp4_morlet` | morlet | 1.0 | on | label-change | Frontend comparison |
| `exp5_interv_strong` | multiscale | 2.0 | on | label-change | Intervention weight sensitivity |

### Loader split
- `build_pair_loaders(data_dir, val_dir, cfg)` → `(DataLoader, DataLoader)` using `ConsecutivePairDataset` + `collate_pairs` — CRL pre-training only
- `build_single_loaders(data_dir, val_dir, cfg)` → `(DataLoader, DataLoader)` using `SensorDataset` + `collate_single` — downstream training and evaluation only

### Full pipeline per experiment
```
train_crl(pair_loaders) → train_downstream(single_loaders) → evaluate
```

### Missing `csv` import fix
Add `import csv` to `run_experiments.py` (currently crashes at line 179 without it).

---

## 5. Metrics Recording & Monitoring

### CRL pre-training CSV (`crl_metrics.csv`)
Extended columns:

| Column | New? | Purpose |
|---|---|---|
| `epoch`, `beta`, `beta_event` | existing/new | Schedule tracking + decision per epoch |
| `lr` | new | Current LR from scheduler |
| `grad_norm` | new | Gradient norm before clipping — instability detection |
| `train/val_recon`, `train/val_kl`, `train/val_interv` | existing | Core ELBO |
| `train/val_aux_pres`, `train/val_aux_type`, `train/val_aux_prox` | new | Auxiliary loss contributions |
| `val_ref_elbo`, `val_raw_kl` | existing | Checkpoint metric + collapse detection |
| `pres_changed_frac`, `type_changed_frac` | new | Fraction of pairs with label change — validates intervention signal health |

### Downstream metrics CSV (`downstream_metrics_{sensor}.csv`)
Extended columns:

| Column | New? | Purpose |
|---|---|---|
| `val_pres_acc`, `val_pres_f1` | existing | Presence probe quality |
| `val_type_acc`, `val_type_f1` | existing | Type probe quality |
| `val_prox_loss` | new | Proximity regression MSE |
| `class_breakdown` | new | Per-class F1 as JSON string — detects class collapse |

### Console output format
```
Epoch  12 | β=0.24(↑) lr=2.8e-4 | recon=0.284 kl=1.71 interv=0.094 | aux_p=0.681 aux_t=0.943 aux_px=0.032 | ref_elbo=2.195 | ‖g‖=0.84 | Δpres=0.31 Δtype=0.18
```

### Per-experiment `experiment_summary.json`
Added fields: `best_pres_f1_audio`, `best_pres_f1_seismic`, `best_type_f1_audio`, `best_type_f1_seismic`, `best_prox_loss`, `crl_converged_epoch`, `downstream_epochs_run`.

### Report verdict logic
Primary signal changed from `val_ref_elbo` to downstream F1:
- `IMPROVED`: mean downstream F1 > baseline + 0.05
- `MARGINAL`: mean downstream F1 > baseline
- `NO_CHANGE`: otherwise

---

## 6. Hardware-Adaptive Configuration

`hardware_profile()` function detects GPU memory at startup and selects a config tier:

| Tier | Condition | `batch_size` | `d_model` | `n_layers` | `num_workers` | `steps_per_epoch` |
|---|---|---|---|---|---|---|
| `h100` | VRAM ≥ 60GB | 512 | 128 | 4 | 12 | None |
| `mid` | VRAM 16–60GB | 128 | 64 | 2 | 8 | None |
| `low` | VRAM 8–16GB | 64 | 64 | 2 | 4 | 200 |
| `cpu` | no CUDA/MPS | 32 | 32 | 1 | 2 | 50 |

`n_heads = max(2, d_model // 32)`. `d_z = 24` fixed across all tiers.

CLI `--hardware-profile {h100,mid,low,cpu}` overrides auto-detection. Manual overrides (`--batch-size`, `--steps-per-epoch`, etc.) take precedence over profile.

Startup print:
```
  Hardware: NVIDIA H100 80GB SXM5 → profile=h100 (batch=512, d_model=128, workers=12)
```

---

## Files to Modify

| File | Change |
|---|---|
| `crl_vehicle/models/latent.py` | Expand `d_z=24`, raw slices in `split()`, add `LinearProximityHead` |
| `crl_vehicle/models/heads.py` | Resize `LinearPresenceHead(4,1)`, `LinearTypeHead(6,n)`, add `LinearProximityHead(3,1)` |
| `crl_vehicle/models/intervention.py` | Redesign classifier: input `z_env` concat, output 2-bit BCE |
| `crl_vehicle/data/dataset.py` | `ConsecutivePairDataset` emits label pairs; `collate_pairs` includes new keys |
| `crl_vehicle/config.py` | Add `d_z`, aux loss weights, `lambda_interv` redesign flag, hardware profile fields |
| `training/trainer.py` | Add aux heads to `CRLModel`; extend loss, metrics logging, downstream loader fix; add `LinearProximityHead` to downstream eval (MSE only, no label supervision yet) |
| `run_experiments.py` | Add `csv` import, downstream pipeline, `build_single_loaders`, new experiment grid, hardware profile, updated report |

---

## Verification

1. Run `python run_experiments.py --steps-per-epoch 50 --only exp1_baseline` — confirm no crash, CSV columns present, downstream runs after CRL
2. Check `pres_changed_frac` and `type_changed_frac` in CSV — should be > 0 and < 1 (non-trivial signal)
3. Check `aux_pres` loss in exp2/exp3 — should decrease over epochs (model is learning presence)
4. Compare `val_pres_f1` between `exp1_baseline` and `exp3_redesigned` — expect meaningful lift in exp3
5. Check `grad_norm` stays below 2.0 throughout — if spiking, lambda weights need reduction
6. Confirm `exp4_morlet` vs `exp3_redesigned` report verdict correctly reflects F1 comparison
