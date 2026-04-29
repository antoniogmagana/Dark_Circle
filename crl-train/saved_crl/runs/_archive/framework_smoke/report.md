# Full diagnostic report — framework_smoke

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 0.26 min

- total epochs recorded: 2

- **best val_ref_elbo:** 8.638558 (epoch 1) → `crl_best.pth`

- **best val_aux_type_f1:** 0.3076 (epoch 1) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 1 | 0.5774 | 0.3164 | 0.5297 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 1 | 0.5479 | 0.3206 | 0.5275 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 1 | 0.6283 | 0.3360 | 0.5098 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 1 | 0.5366 | 0.3147 | 0.5227 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 1 | 0.6311 | 0.3185 | 0.5397 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 1 | 0.6017 | 0.3376 | 0.5307 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.2228 | 0.1373 | 0.1373 | 0.1683 |
| linear_fullz__crl_best | focal | 9,228 | 0.2602 | 0.0076 | 0.0152 | 0.0083 |
| linear_fullz__crl_best | iobt | 3,204 | 0.0893 | 0.3246 | 0.6491 | 0.6801 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.1789 | 0.1325 | 0.1325 | 0.1671 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.2106 | 0.0152 | 0.0304 | 0.0160 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.0684 | 0.3070 | 0.6140 | 0.6506 |
| linear_ztype__crl_best | full | 12,432 | 0.1549 | 0.1442 | 0.1442 | 0.1730 |
| linear_ztype__crl_best | focal | 9,228 | 0.1821 | 0.0101 | 0.0203 | 0.0109 |
| linear_ztype__crl_best | iobt | 3,204 | 0.0611 | 0.3386 | 0.6771 | 0.6915 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.1206 | 0.1502 | 0.1502 | 0.1773 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.1433 | 0.0118 | 0.0236 | 0.0123 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.0434 | 0.3503 | 0.7006 | 0.7053 |
| mlp_ztype__crl_best | full | 12,432 | 0.2177 | 0.3005 | 0.3005 | 0.3211 |
| mlp_ztype__crl_best | focal | 9,228 | 0.2540 | 0.1269 | 0.2538 | 0.2257 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.0884 | 0.3357 | 0.6715 | 0.6264 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.1175 | 0.1455 | 0.1455 | 0.1676 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.1392 | 0.0070 | 0.0140 | 0.0069 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.0443 | 0.3367 | 0.6734 | 0.6816 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.000 | 0.282 | 0.027 | 0.239 |
| linear_fullz__crl_best | focal | 0.000 | 0.000 | 0.030 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.866 | 0.000 | 0.432 |
| linear_fullz__crl_best_aux_type | full | 0.034 | 0.295 | 0.024 | 0.177 |
| linear_fullz__crl_best_aux_type | focal | 0.035 | 0.000 | 0.026 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.854 | 0.000 | 0.374 |
| linear_ztype__crl_best | full | 0.011 | 0.292 | 0.027 | 0.246 |
| linear_ztype__crl_best | focal | 0.011 | 0.000 | 0.030 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.855 | 0.000 | 0.499 |
| linear_ztype__crl_best_aux_type | full | 0.024 | 0.296 | 0.022 | 0.259 |
| linear_ztype__crl_best_aux_type | focal | 0.024 | 0.000 | 0.023 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.851 | 0.000 | 0.551 |
| mlp_ztype__crl_best | full | 0.477 | 0.347 | 0.005 | 0.372 |
| mlp_ztype__crl_best | focal | 0.502 | 0.000 | 0.006 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.795 | 0.000 | 0.547 |
| mlp_ztype__crl_best_aux_type | full | 0.022 | 0.273 | 0.005 | 0.282 |
| mlp_ztype__crl_best_aux_type | focal | 0.022 | 0.000 | 0.006 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.843 | 0.000 | 0.504 |
