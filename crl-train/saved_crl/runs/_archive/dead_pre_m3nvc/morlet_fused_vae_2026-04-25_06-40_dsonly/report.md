# Full diagnostic report — 2026-04-25_06-40-58

## CRL pre-training

- frontend: `morlet_learnable`, d_z=24, d_model=64, n_layers=2

- total epochs recorded: 77

- **best val_ref_elbo:** 621.34042 (epoch 51) → `crl_best.pth`

- **best val_aux_type_f1:** 0.2885 (epoch 12) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 34 | 0.4523 | 0.2495 | 0.4417 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 2 | 0.8356 | 0.2813 | 0.3375 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 7 | 0.4367 | 0.2342 | 0.4149 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 12 | 0.8038 | 0.2864 | 0.4333 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 45 | 0.8014 | 0.2516 | 0.4379 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 4 | 0.8140 | 0.2915 | 0.4233 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.7600 | 0.2370 | 0.2370 | 0.4052 |
| linear_fullz__crl_best | focal | 14,534 | 0.7568 | 0.1793 | 0.1793 | 0.2678 |
| linear_fullz__crl_best | iobt | 936 | 0.7188 | 0.1282 | 0.2564 | 0.2021 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.7708 | 0.2815 | 0.2815 | 0.4079 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.7674 | 0.2193 | 0.2193 | 0.2893 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.7241 | 0.1279 | 0.2557 | 0.2142 |
| linear_ztype__crl_best | full | 21,436 | 0.3951 | 0.2314 | 0.2314 | 0.4124 |
| linear_ztype__crl_best | focal | 14,534 | 0.4101 | 0.1727 | 0.1727 | 0.2733 |
| linear_ztype__crl_best | iobt | 936 | 0.3700 | 0.1354 | 0.2708 | 0.2085 |
| linear_ztype__crl_best_aux_type | full | 21,436 | 0.7975 | 0.2306 | 0.2306 | 0.2785 |
| linear_ztype__crl_best_aux_type | focal | 14,534 | 0.7884 | 0.1767 | 0.1767 | 0.1992 |
| linear_ztype__crl_best_aux_type | iobt | 936 | 0.7735 | 0.1710 | 0.3420 | 0.4509 |
| mlp_ztype__crl_best | full | 21,436 | 0.3820 | 0.2176 | 0.2176 | 0.3921 |
| mlp_ztype__crl_best | focal | 14,534 | 0.3970 | 0.1636 | 0.1636 | 0.2604 |
| mlp_ztype__crl_best | iobt | 936 | 0.3595 | 0.1281 | 0.2562 | 0.2182 |
| mlp_ztype__crl_best_aux_type | full | 21,436 | 0.7602 | 0.2687 | 0.2687 | 0.3942 |
| mlp_ztype__crl_best_aux_type | focal | 14,534 | 0.7586 | 0.1878 | 0.1878 | 0.2732 |
| mlp_ztype__crl_best_aux_type | iobt | 936 | 0.7117 | 0.1451 | 0.2901 | 0.2287 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.011 | 0.095 | 0.586 | 0.256 |
| linear_fullz__crl_best | focal | 0.011 | 0.108 | 0.434 | 0.164 |
| linear_fullz__crl_best | iobt | 0.000 | 0.003 | 0.000 | 0.510 |
| linear_fullz__crl_best_aux_type | full | 0.044 | 0.229 | 0.588 | 0.266 |
| linear_fullz__crl_best_aux_type | focal | 0.046 | 0.257 | 0.445 | 0.129 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.000 | 0.000 | 0.511 |
| linear_ztype__crl_best | full | 0.009 | 0.051 | 0.591 | 0.274 |
| linear_ztype__crl_best | focal | 0.009 | 0.055 | 0.442 | 0.184 |
| linear_ztype__crl_best | iobt | 0.000 | 0.024 | 0.000 | 0.518 |
| linear_ztype__crl_best_aux_type | full | 0.048 | 0.212 | 0.297 | 0.365 |
| linear_ztype__crl_best_aux_type | focal | 0.050 | 0.238 | 0.138 | 0.281 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.021 | 0.000 | 0.663 |
| mlp_ztype__crl_best | full | 0.003 | 0.001 | 0.567 | 0.300 |
| mlp_ztype__crl_best | focal | 0.003 | 0.001 | 0.415 | 0.236 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.003 | 0.000 | 0.509 |
| mlp_ztype__crl_best_aux_type | full | 0.030 | 0.216 | 0.576 | 0.254 |
| mlp_ztype__crl_best_aux_type | focal | 0.030 | 0.245 | 0.424 | 0.051 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.060 | 0.000 | 0.520 |
