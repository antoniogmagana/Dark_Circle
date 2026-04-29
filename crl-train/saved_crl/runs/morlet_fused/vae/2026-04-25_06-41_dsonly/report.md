# Full diagnostic report — 2026-04-25_06-41-49

## CRL pre-training

- frontend: `morlet_learnable`, d_z=24, d_model=64, n_layers=2

- total epochs recorded: 73

- **best val_ref_elbo:** 621.33655 (epoch 47) → `crl_best.pth`

- **best val_aux_type_f1:** 0.2868 (epoch 7) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 27 | 0.7961 | 0.2563 | 0.4464 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 1 | 0.8478 | 0.2821 | 0.3296 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 8 | 0.7976 | 0.2760 | 0.4597 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 3 | 0.4892 | 0.2975 | 0.4383 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 41 | 0.4324 | 0.2639 | 0.4382 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 28 | 0.8031 | 0.2909 | 0.4243 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.3785 | 0.2503 | 0.2503 | 0.4065 |
| linear_fullz__crl_best | focal | 14,534 | 0.3978 | 0.1900 | 0.1900 | 0.2692 |
| linear_fullz__crl_best | iobt | 936 | 0.3335 | 0.1283 | 0.2566 | 0.2053 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.7611 | 0.2788 | 0.2788 | 0.4063 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.7568 | 0.2166 | 0.2166 | 0.2877 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.7156 | 0.1278 | 0.2556 | 0.2182 |
| linear_ztype__crl_best | full | 21,436 | 0.7552 | 0.2367 | 0.2367 | 0.4107 |
| linear_ztype__crl_best | focal | 14,534 | 0.7545 | 0.1739 | 0.1739 | 0.2678 |
| linear_ztype__crl_best | iobt | 936 | 0.7079 | 0.1373 | 0.2747 | 0.2110 |
| linear_ztype__crl_best_aux_type | full | 21,436 | 0.8102 | 0.2396 | 0.2396 | 0.2636 |
| linear_ztype__crl_best_aux_type | focal | 14,534 | 0.8039 | 0.1851 | 0.1851 | 0.2180 |
| linear_ztype__crl_best_aux_type | iobt | 936 | 0.7691 | 0.2576 | 0.5151 | 0.4847 |
| mlp_ztype__crl_best | full | 21,436 | 0.7565 | 0.2433 | 0.2433 | 0.4039 |
| mlp_ztype__crl_best | focal | 14,534 | 0.7560 | 0.1584 | 0.1584 | 0.2628 |
| mlp_ztype__crl_best | iobt | 936 | 0.7084 | 0.1805 | 0.3611 | 0.2399 |
| mlp_ztype__crl_best_aux_type | full | 21,436 | 0.4311 | 0.2734 | 0.2734 | 0.3992 |
| mlp_ztype__crl_best_aux_type | focal | 14,534 | 0.4413 | 0.1990 | 0.1990 | 0.2753 |
| mlp_ztype__crl_best_aux_type | iobt | 936 | 0.4106 | 0.1351 | 0.2703 | 0.2262 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.012 | 0.155 | 0.587 | 0.248 |
| linear_fullz__crl_best | focal | 0.012 | 0.171 | 0.432 | 0.145 |
| linear_fullz__crl_best | iobt | 0.000 | 0.021 | 0.000 | 0.493 |
| linear_fullz__crl_best_aux_type | full | 0.037 | 0.228 | 0.586 | 0.264 |
| linear_fullz__crl_best_aux_type | focal | 0.038 | 0.256 | 0.446 | 0.126 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.000 | 0.000 | 0.511 |
| linear_ztype__crl_best | full | 0.008 | 0.079 | 0.588 | 0.272 |
| linear_ztype__crl_best | focal | 0.008 | 0.083 | 0.433 | 0.172 |
| linear_ztype__crl_best | iobt | 0.000 | 0.049 | 0.000 | 0.500 |
| linear_ztype__crl_best_aux_type | full | 0.081 | 0.263 | 0.333 | 0.281 |
| linear_ztype__crl_best_aux_type | focal | 0.086 | 0.318 | 0.187 | 0.150 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.520 | 0.000 | 0.511 |
| mlp_ztype__crl_best | full | 0.005 | 0.174 | 0.595 | 0.200 |
| mlp_ztype__crl_best | focal | 0.005 | 0.181 | 0.429 | 0.018 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.246 | 0.000 | 0.476 |
| mlp_ztype__crl_best_aux_type | full | 0.038 | 0.204 | 0.577 | 0.274 |
| mlp_ztype__crl_best_aux_type | focal | 0.039 | 0.229 | 0.428 | 0.099 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.026 | 0.000 | 0.514 |
