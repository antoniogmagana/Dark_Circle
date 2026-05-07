# Full diagnostic report — 2026-04-25_10-06-01

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 662.81 min

- total epochs recorded: 96

- **best val_ref_elbo:** 1.976477 (epoch 70) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4661 (epoch 71) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 41 | 0.6308 | 0.3576 | 0.5041 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 43 | 0.6330 | 0.3483 | 0.5070 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 39 | 0.7643 | 0.3548 | 0.4361 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 19 | 0.5941 | 0.3381 | 0.4310 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.7658 | 0.3537 | 0.3537 | 0.4295 |
| linear_fullz__crl_best | focal | 14,534 | 0.7915 | 0.2535 | 0.2535 | 0.2806 |
| linear_fullz__crl_best | iobt | 936 | 0.7468 | 0.1736 | 0.3473 | 0.2778 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.6217 | 0.3588 | 0.3588 | 0.4520 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.6328 | 0.2607 | 0.2607 | 0.3067 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.6830 | 0.1738 | 0.3477 | 0.2681 |
| linear_signal__crl_best | full | 21,436 | 0.6559 | 0.2791 | 0.2791 | 0.4467 |
| linear_signal__crl_best | focal | 14,534 | 0.6847 | 0.1668 | 0.1668 | 0.2641 |
| linear_signal__crl_best | iobt | 936 | 0.6323 | 0.2700 | 0.5400 | 0.4919 |
| linear_signal__crl_best_aux_type | full | 21,436 | 0.6527 | 0.2775 | 0.2775 | 0.4695 |
| linear_signal__crl_best_aux_type | focal | 14,534 | 0.6609 | 0.1604 | 0.1604 | 0.2956 |
| linear_signal__crl_best_aux_type | iobt | 936 | 0.7050 | 0.2847 | 0.5695 | 0.4726 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.361 | 0.211 | 0.595 | 0.248 |
| linear_fullz__crl_best | focal | 0.372 | 0.215 | 0.325 | 0.102 |
| linear_fullz__crl_best | iobt | 0.000 | 0.177 | 0.000 | 0.517 |
| linear_fullz__crl_best_aux_type | full | 0.364 | 0.211 | 0.620 | 0.240 |
| linear_fullz__crl_best_aux_type | focal | 0.375 | 0.214 | 0.380 | 0.073 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.180 | 0.000 | 0.515 |
| linear_signal__crl_best | full | 0.004 | 0.159 | 0.605 | 0.348 |
| linear_signal__crl_best | focal | 0.004 | 0.117 | 0.419 | 0.126 |
| linear_signal__crl_best | iobt | 0.000 | 0.385 | 0.000 | 0.695 |
| linear_signal__crl_best_aux_type | full | 0.005 | 0.141 | 0.626 | 0.337 |
| linear_signal__crl_best_aux_type | focal | 0.005 | 0.085 | 0.464 | 0.088 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.431 | 0.000 | 0.708 |
