# Full diagnostic report — 2026-04-22_06-37-09

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 131.35 min

- total epochs recorded: 84

- **best val_ref_elbo:** 1.990382 (epoch 58) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4006 (epoch 17) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 25 | 0.7596 | 0.3514 | 0.5806 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 31 | 0.6747 | 0.4758 | 0.5943 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 17 | 0.7452 | 0.4897 | 0.6180 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 33 | 0.6915 | 0.4608 | 0.5881 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 7 | 0.7497 | 0.4022 | 0.6069 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 36 | 0.6738 | 0.4513 | 0.6125 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_f1 | type_f1_support_only | pres_f1_cal | type_f1_cal | type_f1_support_only_cal | type_acc |
|---|---|---|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7170 | 0.2739 | 0.2739 | 0.8133 | 0.2522 | 0.2522 | 0.3695 |
| linear_fullz__crl_best | focal | 9,228 | 0.7172 | 0.2272 | 0.4545 | 0.8274 | 0.2914 | 0.5827 | 0.4067 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7164 | 0.1303 | 0.2606 | 0.7776 | 0.2059 | 0.4117 | 0.2504 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.6272 | 0.2484 | 0.2484 | 0.8118 | 0.2291 | 0.2291 | 0.3112 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.6351 | 0.2126 | 0.4252 | 0.8241 | 0.2659 | 0.5319 | 0.3650 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.6024 | 0.1056 | 0.2112 | 0.7858 | 0.2203 | 0.4405 | 0.1387 |
| linear_ztype__crl_best | full | 12,432 | 0.7337 | 0.2413 | 0.2413 | 0.8136 | 0.2387 | 0.2387 | 0.3492 |
| linear_ztype__crl_best | focal | 9,228 | 0.7328 | 0.2481 | 0.4962 | 0.8274 | 0.2759 | 0.5517 | 0.4432 |
| linear_ztype__crl_best | iobt | 3,204 | 0.7366 | 0.0473 | 0.0945 | 0.7777 | 0.2059 | 0.4117 | 0.0485 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.6285 | 0.2067 | 0.2067 | 0.8120 | 0.2300 | 0.2300 | 0.2285 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.6363 | 0.1571 | 0.3142 | 0.8241 | 0.2661 | 0.5322 | 0.2426 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.6040 | 0.1443 | 0.2886 | 0.7859 | 0.2102 | 0.4204 | 0.1833 |
| mlp_ztype__crl_best | full | 12,432 | 0.7012 | 0.1411 | 0.1411 | 0.8133 | 0.2266 | 0.2266 | 0.1851 |
| mlp_ztype__crl_best | focal | 9,228 | 0.7010 | 0.1110 | 0.2220 | 0.8273 | 0.2620 | 0.5239 | 0.1457 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7019 | 0.1399 | 0.2798 | 0.7777 | 0.2059 | 0.4117 | 0.3110 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.6530 | 0.1534 | 0.1534 | 0.8126 | 0.2150 | 0.2150 | 0.2036 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.6584 | 0.1379 | 0.2759 | 0.8247 | 0.2514 | 0.5028 | 0.2108 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.6361 | 0.1026 | 0.2051 | 0.7829 | 0.2409 | 0.4818 | 0.1806 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.318 | 0.236 | 0.494 | 0.048 |
| linear_fullz__crl_best | focal | 0.336 | 0.000 | 0.573 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.470 | 0.000 | 0.051 |
| linear_fullz__crl_best_aux_type | full | 0.415 | 0.130 | 0.321 | 0.128 |
| linear_fullz__crl_best_aux_type | focal | 0.469 | 0.000 | 0.381 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.281 | 0.000 | 0.142 |
| linear_ztype__crl_best | full | 0.486 | 0.044 | 0.380 | 0.055 |
| linear_ztype__crl_best | focal | 0.550 | 0.000 | 0.443 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.047 | 0.000 | 0.142 |
| linear_ztype__crl_best_aux_type | full | 0.313 | 0.118 | 0.232 | 0.164 |
| linear_ztype__crl_best_aux_type | focal | 0.355 | 0.000 | 0.274 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.295 | 0.000 | 0.282 |
| mlp_ztype__crl_best | full | 0.101 | 0.182 | 0.281 | 0.000 |
| mlp_ztype__crl_best | focal | 0.111 | 0.000 | 0.333 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.560 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.301 | 0.137 | 0.169 | 0.006 |
| mlp_ztype__crl_best_aux_type | focal | 0.352 | 0.000 | 0.200 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.404 | 0.000 | 0.006 |
