# Full diagnostic report — 2026-04-24_18-57-29

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 81.58 min

- total epochs recorded: 61

- **best val_ref_elbo:** 621.326776 (epoch 35) → `crl_best.pth`

- **best val_aux_type_f1:** 0.2837 (epoch 10) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 36 | 0.8006 | 0.2538 | 0.4359 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 31 | 0.4724 | 0.3017 | 0.4256 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 9 | 0.8007 | 0.2408 | 0.3732 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 16 | 0.8263 | 0.2919 | 0.4268 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 30 | 0.8009 | 0.2608 | 0.4208 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 47 | 0.4599 | 0.2900 | 0.4088 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.7561 | 0.2357 | 0.2357 | 0.3729 |
| linear_fullz__crl_best | focal | 14,534 | 0.7567 | 0.1761 | 0.1761 | 0.2444 |
| linear_fullz__crl_best | iobt | 936 | 0.7001 | 0.1263 | 0.2526 | 0.2158 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.3954 | 0.2877 | 0.2877 | 0.4119 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.4148 | 0.2294 | 0.2294 | 0.3076 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.3571 | 0.1303 | 0.2606 | 0.2230 |
| linear_ztype__crl_best | full | 21,436 | 0.7557 | 0.2136 | 0.2136 | 0.3747 |
| linear_ztype__crl_best | focal | 14,534 | 0.7562 | 0.1469 | 0.1469 | 0.2337 |
| linear_ztype__crl_best | iobt | 936 | 0.6993 | 0.1291 | 0.2581 | 0.2174 |
| linear_ztype__crl_best_aux_type | full | 21,436 | 0.4113 | 0.2745 | 0.2745 | 0.3961 |
| linear_ztype__crl_best_aux_type | focal | 14,534 | 0.4292 | 0.2113 | 0.2113 | 0.2849 |
| linear_ztype__crl_best_aux_type | iobt | 936 | 0.3741 | 0.1309 | 0.2619 | 0.2238 |
| mlp_ztype__crl_best | full | 21,436 | 0.7558 | 0.2310 | 0.2310 | 0.3563 |
| mlp_ztype__crl_best | focal | 14,534 | 0.7564 | 0.1878 | 0.1878 | 0.2511 |
| mlp_ztype__crl_best | iobt | 936 | 0.7001 | 0.1249 | 0.2497 | 0.2295 |
| mlp_ztype__crl_best_aux_type | full | 21,436 | 0.7842 | 0.2683 | 0.2683 | 0.3909 |
| mlp_ztype__crl_best_aux_type | focal | 14,534 | 0.7793 | 0.1938 | 0.1938 | 0.2794 |
| mlp_ztype__crl_best_aux_type | iobt | 936 | 0.7413 | 0.1371 | 0.2742 | 0.2279 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.016 | 0.162 | 0.548 | 0.218 |
| linear_fullz__crl_best | focal | 0.016 | 0.186 | 0.396 | 0.106 |
| linear_fullz__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.505 |
| linear_fullz__crl_best_aux_type | full | 0.058 | 0.250 | 0.594 | 0.250 |
| linear_fullz__crl_best_aux_type | focal | 0.061 | 0.280 | 0.472 | 0.105 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.000 | 0.000 | 0.521 |
| linear_ztype__crl_best | full | 0.014 | 0.058 | 0.543 | 0.239 |
| linear_ztype__crl_best | focal | 0.014 | 0.067 | 0.385 | 0.122 |
| linear_ztype__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.516 |
| linear_ztype__crl_best_aux_type | full | 0.059 | 0.206 | 0.573 | 0.260 |
| linear_ztype__crl_best_aux_type | focal | 0.063 | 0.231 | 0.440 | 0.111 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.006 | 0.000 | 0.518 |
| mlp_ztype__crl_best | full | 0.008 | 0.113 | 0.522 | 0.281 |
| mlp_ztype__crl_best | focal | 0.008 | 0.128 | 0.397 | 0.217 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.003 | 0.000 | 0.496 |
| mlp_ztype__crl_best_aux_type | full | 0.040 | 0.213 | 0.571 | 0.250 |
| mlp_ztype__crl_best_aux_type | focal | 0.042 | 0.241 | 0.434 | 0.058 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.029 | 0.000 | 0.519 |
