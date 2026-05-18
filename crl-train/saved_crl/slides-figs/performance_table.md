# Multiscale run performance — completed full pipelines (CRL + downstream + eval)

Source: each run's `report.json` `evals` list, `split == "full"`, no dataset filter. Morlet rows have been removed; only multiscale runs are reported here.

## Run inventory

| Run dir | Status | Frontend | Training mode | use_id_split | Test n_windows |
|---|---|---|---|---|---:|
| `multiscale_run1_diag` (legacy) | complete | multiscale | vae (legacy 4-block) | **False** | 17,123 |
| `multiscale_v2` (legacy) | complete | multiscale | vae (legacy 4-block) | True | 91,325 |
| `multiscale/vae/2026-05-03_05-02-44` | complete | multiscale | vae | True | rerun |
| `multiscale/disentangled/2026-05-03_05-03-14` | complete | multiscale | disentangled (signal/env) | True | rerun |

> **Caveat — different test splits.** `multiscale_run1_diag` ran with `use_id_split=False` and was evaluated on **17,123 windows**; `multiscale_v2` evaluated on **91,325 windows** under the ID-split schema. F1 numbers are not directly comparable across this split boundary.

> **Caveat — training mode.** The two legacy completed runs use the **legacy `vae` mode** (4-block causal latent). The current comparison set uses the modern `vae` and `disentangled` modes; see `_summary.json` for live numbers.

---

## Headline (per legacy run, by `meta.json` probe × ckpt)

The "headline" probe and checkpoint matching each run's `meta.json`:

| Run | Probe × ckpt | Pres F1 | Pres acc | Type macro F1 | Type acc |
|---|---|---:|---:|---:|---:|
| `multiscale_run1_diag` | linear_ztype × crl_best | 0.795 | 0.737 | 0.442 | 0.453 |
| `multiscale_v2` | linear_ztype × crl_best | **0.871** | **0.809** | **0.527** | **0.664** |

`multiscale_v2` wins on every metric on the larger 91,325-window split.

---

## Full table — all 6 probes × ckpts (legacy multiscale runs)

Probe-mode key: `linear_ztype` (default) reads `z_type` only (D_TYPE=6 dims); `linear_fullz` reads the full latent (24 dims); `mlp_ztype` is a small MLP on `z_type`.
Checkpoint key: `crl_best.pth` (selected by `val_ref_elbo`); `crl_best_aux_type.pth` (selected by `val_aux_type_f1`, downstream-proxy).

| Run | Probe | Ckpt | Pres F1 | Pres acc | Type macro F1 | Type acc |
|---|---|---|---:|---:|---:|---:|
| multiscale_run1_diag (n=17,123) | linear_fullz | crl_best | 0.796 | 0.738 | 0.450 | 0.459 |
| multiscale_run1_diag | linear_fullz | crl_best_aux_type | 0.796 | 0.742 | 0.444 | 0.440 |
| multiscale_run1_diag | linear_ztype | crl_best | 0.795 | 0.737 | 0.442 | 0.453 |
| multiscale_run1_diag | linear_ztype | crl_best_aux_type | 0.794 | 0.741 | 0.444 | 0.440 |
| multiscale_run1_diag | mlp_ztype | crl_best | 0.796 | 0.738 | 0.439 | 0.446 |
| multiscale_run1_diag | mlp_ztype | crl_best_aux_type | 0.794 | 0.741 | 0.454 | 0.452 |
| **multiscale_v2 (n=91,325)** | linear_fullz | crl_best | **0.873** | **0.812** | 0.527 | 0.667 |
| multiscale_v2 | linear_fullz | crl_best_aux_type | 0.854 | 0.788 | **0.551** | 0.647 |
| multiscale_v2 | linear_ztype | crl_best | 0.871 | 0.809 | 0.527 | 0.664 |
| multiscale_v2 | linear_ztype | crl_best_aux_type | 0.855 | 0.789 | 0.551 | 0.648 |
| multiscale_v2 | mlp_ztype | crl_best | 0.864 | 0.802 | 0.516 | **0.686** |
| multiscale_v2 | mlp_ztype | crl_best_aux_type | 0.855 | 0.789 | 0.546 | 0.664 |

### Observations

- For **`multiscale_v2`**, `crl_best.pth` (β-invariant ELBO selector) gives slightly better presence F1; `crl_best_aux_type.pth` gives slightly better type macro F1. The dual-checkpoint design trades one for the other as expected.
- **`linear_fullz` and `linear_ztype` are within ~0.5 F1 points** on `multiscale_v2`, suggesting most of the type information has localized into `z_type` already (the `vae` mode's intent).
