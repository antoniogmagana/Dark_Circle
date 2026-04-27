# id_split run performance — completed full pipelines (CRL + downstream + eval)

Source: each run's `report.json` `evals` list, `split == "full"`, no dataset filter.
Generated 2026-04-27 from `crl-train/saved_crl/id_split/`.

## Run inventory

| Run dir | Status | Frontend | Training mode | use_id_split | Test n_windows |
|---|---|---|---|---|---:|
| `multiscale_run1_diag` | complete | multiscale | vae (legacy 4-block) | **False** | 17,123 |
| `morlet_per_sensor_phase_run1_diag` | complete | morlet_per_sensor (+ phase) | vae (legacy 4-block) | True | 91,325 |
| `multiscale_v2` | complete | multiscale | vae (legacy 4-block) | True | 91,325 |
| `disentangled_multiscale_run1` | **in progress** (epoch 82 of 100, CRL only) | multiscale | disentangled (signal/env) | — | — |
| `multiscale_run1` | partial (CRL + downstream, no eval) | multiscale | vae | False | — |
| `morlet_per_sensor_phase_run1` | partial (CRL + downstream, no eval) | morlet_per_sensor | vae | False | — |

> **Caveat — different test splits.** `multiscale_run1_diag` ran with `use_id_split=False` and was evaluated on **17,123 windows**; the other two evaluated on **91,325 windows** under the ID-split schema. F1 numbers are not directly comparable across this split boundary. Treat `multiscale_run1_diag` as a separate point of reference, not a head-to-head against the other two.

> **Caveat — training mode.** All three completed runs use the **legacy `vae` mode** (4-block causal latent: pres / type / prox / env / free). The disentangled `signal / env` two-block mode is still in progress (`disentangled_multiscale_run1`).

---

## Headline (per run, by `meta.json` probe × ckpt)

The "headline" probe and checkpoint matching each run's `meta.json`:

| Run | Probe × ckpt | Pres F1 | Pres acc | Type macro F1 | Type acc |
|---|---|---:|---:|---:|---:|
| `multiscale_run1_diag` | linear_ztype × crl_best | 0.795 | 0.737 | 0.442 | 0.453 |
| `morlet_per_sensor_phase_run1_diag` | linear_ztype × crl_best | 0.723 | 0.648 | 0.445 | 0.583 |
| `multiscale_v2` | linear_ztype × crl_best | **0.871** | **0.809** | **0.527** | **0.664** |

**`multiscale_v2` wins on every metric** even after accounting for the smaller-test-set caveat on `multiscale_run1_diag`. Compared with `morlet_per_sensor_phase_run1_diag` (same test set, same `linear_ztype × crl_best`):
- Pres F1: **+14.8 pts** (0.871 vs 0.723)
- Type macro F1: **+8.2 pts** (0.527 vs 0.445)
- Type acc: **+8.1 pts** (0.664 vs 0.583)

---

## Full table — all 6 probes × ckpts × 3 runs

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
| morlet_per_sensor_phase_run1_diag (n=91,325) | linear_fullz | crl_best | 0.782 | 0.697 | 0.397 | 0.537 |
| morlet_per_sensor_phase_run1_diag | linear_fullz | crl_best_aux_type | 0.779 | 0.698 | 0.414 | 0.530 |
| morlet_per_sensor_phase_run1_diag | linear_ztype | crl_best | 0.723 | 0.648 | 0.445 | 0.583 |
| morlet_per_sensor_phase_run1_diag | linear_ztype | crl_best_aux_type | 0.780 | 0.699 | 0.484 | 0.608 |
| morlet_per_sensor_phase_run1_diag | mlp_ztype | crl_best | 0.771 | 0.689 | 0.411 | 0.575 |
| morlet_per_sensor_phase_run1_diag | mlp_ztype | crl_best_aux_type | 0.776 | 0.694 | 0.478 | 0.610 |
| **multiscale_v2 (n=91,325)** | linear_fullz | crl_best | **0.873** | **0.812** | 0.527 | 0.667 |
| multiscale_v2 | linear_fullz | crl_best_aux_type | 0.854 | 0.788 | **0.551** | 0.647 |
| multiscale_v2 | linear_ztype | crl_best | 0.871 | 0.809 | 0.527 | 0.664 |
| multiscale_v2 | linear_ztype | crl_best_aux_type | 0.855 | 0.789 | 0.551 | 0.648 |
| multiscale_v2 | mlp_ztype | crl_best | 0.864 | 0.802 | 0.516 | **0.686** |
| multiscale_v2 | mlp_ztype | crl_best_aux_type | 0.855 | 0.789 | 0.546 | 0.664 |

### Observations

- For **`multiscale_v2`**, `crl_best.pth` (β-invariant ELBO selector) gives slightly better presence F1; `crl_best_aux_type.pth` gives slightly better type macro F1. The dual-checkpoint design trades one for the other as expected.
- **`linear_fullz` and `linear_ztype` are within ~0.5 F1 points** on `multiscale_v2`, suggesting most of the type information has localized into `z_type` already (the `vae` mode's intent).
- For **morlet_per_sensor**, the `crl_best_aux_type` checkpoint is clearly preferable for type discrimination (type macro F1 0.484 vs 0.445 with `linear_ztype` — the +3.9-pt swing the dual-checkpoint design exists to provide).
