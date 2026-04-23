# Full diagnostic report — 2026-04-23_04-10-16

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 174.67 min

- total epochs recorded: 92

- **best val_ref_elbo:** 720.601033 (epoch 66) → `crl_best.pth`

- **best val_aux_type_f1:** 0.3955 (epoch 26) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 16 | 0.5101 | 0.4105 | 0.5332 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 45 | 0.7805 | 0.4031 | 0.5381 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 3 | 0.4963 | 0.3639 | 0.4915 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 8 | 0.4565 | 0.3741 | 0.5034 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 24 | 0.4856 | 0.4129 | 0.5250 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 2 | 0.4707 | 0.4012 | 0.5444 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.3204 | 0.1419 | 0.1419 | 0.2402 |
| linear_fullz__crl_best | focal | 9,228 | 0.2682 | 0.1496 | 0.2992 | 0.3013 |
| linear_fullz__crl_best | iobt | 3,204 | 0.4566 | 0.0367 | 0.0735 | 0.0448 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.4024 | 0.2063 | 0.2063 | 0.2915 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.3583 | 0.1792 | 0.3584 | 0.3449 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.5175 | 0.0979 | 0.1958 | 0.1205 |
| linear_ztype__crl_best | full | 12,432 | 0.3609 | 0.1365 | 0.1365 | 0.2400 |
| linear_ztype__crl_best | focal | 9,228 | 0.3165 | 0.1453 | 0.2905 | 0.3014 |
| linear_ztype__crl_best | iobt | 3,204 | 0.4784 | 0.0364 | 0.0728 | 0.0436 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7262 | 0.1646 | 0.1646 | 0.2694 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7136 | 0.1638 | 0.3276 | 0.3299 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.7600 | 0.0630 | 0.1260 | 0.0759 |
| mlp_ztype__crl_best | full | 12,432 | 0.3380 | 0.2021 | 0.2021 | 0.2881 |
| mlp_ztype__crl_best | focal | 9,228 | 0.2897 | 0.1581 | 0.3162 | 0.3259 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.4649 | 0.1258 | 0.2516 | 0.1673 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.3899 | 0.1856 | 0.1856 | 0.2752 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.3416 | 0.1628 | 0.3256 | 0.3253 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.5146 | 0.0966 | 0.1933 | 0.1148 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.058 | 0.021 | 0.448 | 0.040 |
| linear_fullz__crl_best | focal | 0.058 | 0.000 | 0.540 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.030 | 0.000 | 0.117 |
| linear_fullz__crl_best_aux_type | full | 0.154 | 0.090 | 0.463 | 0.117 |
| linear_fullz__crl_best_aux_type | focal | 0.157 | 0.000 | 0.560 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.181 | 0.000 | 0.211 |
| linear_ztype__crl_best | full | 0.035 | 0.018 | 0.453 | 0.040 |
| linear_ztype__crl_best | focal | 0.035 | 0.000 | 0.546 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.024 | 0.000 | 0.121 |
| linear_ztype__crl_best_aux_type | full | 0.088 | 0.004 | 0.474 | 0.092 |
| linear_ztype__crl_best_aux_type | focal | 0.090 | 0.000 | 0.565 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.006 | 0.000 | 0.246 |
| mlp_ztype__crl_best | full | 0.057 | 0.128 | 0.475 | 0.148 |
| mlp_ztype__crl_best | focal | 0.057 | 0.000 | 0.575 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.285 | 0.000 | 0.218 |
| mlp_ztype__crl_best_aux_type | full | 0.096 | 0.048 | 0.463 | 0.136 |
| mlp_ztype__crl_best_aux_type | focal | 0.097 | 0.000 | 0.554 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.083 | 0.000 | 0.304 |
