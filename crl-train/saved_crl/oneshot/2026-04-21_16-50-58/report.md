# Full diagnostic report — 2026-04-21_16-50-58

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 138.39 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.241628 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.7009 (epoch 28) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 31 | 0.8705 | 0.6719 | 0.7179 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 38 | 0.8706 | 0.6870 | 0.7256 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 49 | 0.8724 | 0.6811 | 0.7417 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 14 | 0.8705 | 0.6937 | 0.7440 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 29 | 0.8731 | 0.6721 | 0.7154 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 49 | 0.8711 | 0.6864 | 0.7251 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7740 | 0.2146 | 0.2146 | 0.2158 |
| linear_fullz__crl_best | focal | 9,228 | 0.7788 | 0.1064 | 0.2129 | 0.1564 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7607 | 0.2264 | 0.4528 | 0.4061 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.7888 | 0.2102 | 0.2102 | 0.2076 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.7960 | 0.1046 | 0.2093 | 0.1488 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.7681 | 0.2140 | 0.4279 | 0.3958 |
| linear_ztype__crl_best | full | 12,432 | 0.7662 | 0.2218 | 0.2218 | 0.2256 |
| linear_ztype__crl_best | focal | 9,228 | 0.7697 | 0.1119 | 0.2239 | 0.1662 |
| linear_ztype__crl_best | iobt | 3,204 | 0.7566 | 0.2330 | 0.4661 | 0.4155 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7874 | 0.2125 | 0.2125 | 0.2088 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7944 | 0.1042 | 0.2084 | 0.1476 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.7674 | 0.2198 | 0.4395 | 0.4046 |
| mlp_ztype__crl_best | full | 12,432 | 0.7704 | 0.2435 | 0.2435 | 0.2610 |
| mlp_ztype__crl_best | focal | 9,228 | 0.7747 | 0.1420 | 0.2840 | 0.2385 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7585 | 0.1992 | 0.3984 | 0.3332 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.7874 | 0.2165 | 0.2165 | 0.2265 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.7944 | 0.1237 | 0.2474 | 0.1855 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.7674 | 0.1926 | 0.3851 | 0.3578 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.091 | 0.175 | 0.312 | 0.280 |
| linear_fullz__crl_best | focal | 0.094 | 0.000 | 0.332 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.312 | 0.000 | 0.594 |
| linear_fullz__crl_best_aux_type | full | 0.114 | 0.128 | 0.285 | 0.314 |
| linear_fullz__crl_best_aux_type | focal | 0.119 | 0.000 | 0.300 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.247 | 0.000 | 0.609 |
| linear_ztype__crl_best | full | 0.088 | 0.177 | 0.335 | 0.288 |
| linear_ztype__crl_best | focal | 0.091 | 0.000 | 0.357 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.332 | 0.000 | 0.601 |
| linear_ztype__crl_best_aux_type | full | 0.115 | 0.134 | 0.282 | 0.319 |
| linear_ztype__crl_best_aux_type | focal | 0.120 | 0.000 | 0.297 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.266 | 0.000 | 0.613 |
| mlp_ztype__crl_best | full | 0.126 | 0.129 | 0.400 | 0.320 |
| mlp_ztype__crl_best | focal | 0.132 | 0.000 | 0.436 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.182 | 0.000 | 0.615 |
| mlp_ztype__crl_best_aux_type | full | 0.117 | 0.098 | 0.349 | 0.302 |
| mlp_ztype__crl_best_aux_type | focal | 0.122 | 0.000 | 0.373 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.162 | 0.000 | 0.609 |
