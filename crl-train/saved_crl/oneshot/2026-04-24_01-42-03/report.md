# Full diagnostic report — 2026-04-24_01-42-03

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 162.68 min

- total epochs recorded: 100

- **best val_ref_elbo:** 686.584002 (epoch 75) → `crl_best.pth`

- **best val_aux_type_f1:** 0.3804 (epoch 21) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 41 | 0.4807 | 0.4232 | 0.5532 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 25 | 0.8125 | 0.4256 | 0.5769 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 2 | 0.5982 | 0.3843 | 0.5412 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 5 | 0.5076 | 0.4483 | 0.5783 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 38 | 0.4380 | 0.4171 | 0.5521 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 17 | 0.8170 | 0.4258 | 0.5750 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.2670 | 0.1397 | 0.1397 | 0.2485 |
| linear_fullz__crl_best | focal | 9,228 | 0.1963 | 0.1545 | 0.3090 | 0.3171 |
| linear_fullz__crl_best | iobt | 3,204 | 0.4440 | 0.0253 | 0.0505 | 0.0288 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.7333 | 0.2298 | 0.2298 | 0.3423 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.7286 | 0.2213 | 0.4425 | 0.4274 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.7465 | 0.0716 | 0.1431 | 0.0702 |
| linear_ztype__crl_best | full | 12,432 | 0.3275 | 0.1424 | 0.1424 | 0.2496 |
| linear_ztype__crl_best | focal | 9,228 | 0.2712 | 0.1544 | 0.3087 | 0.3164 |
| linear_ztype__crl_best | iobt | 3,204 | 0.4713 | 0.0314 | 0.0628 | 0.0360 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7245 | 0.2281 | 0.2281 | 0.3405 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7182 | 0.2200 | 0.4399 | 0.4256 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.7420 | 0.0693 | 0.1387 | 0.0683 |
| mlp_ztype__crl_best | full | 12,432 | 0.5782 | 0.1704 | 0.1704 | 0.2795 |
| mlp_ztype__crl_best | focal | 9,228 | 0.5845 | 0.1684 | 0.3368 | 0.3445 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.5588 | 0.0644 | 0.1288 | 0.0717 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.3546 | 0.3190 | 0.3190 | 0.3705 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.3177 | 0.2158 | 0.4316 | 0.4314 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.4549 | 0.1650 | 0.3301 | 0.1757 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.053 | 0.023 | 0.461 | 0.022 |
| linear_fullz__crl_best | focal | 0.053 | 0.000 | 0.565 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.032 | 0.000 | 0.069 |
| linear_fullz__crl_best_aux_type | full | 0.268 | 0.017 | 0.502 | 0.132 |
| linear_fullz__crl_best_aux_type | focal | 0.274 | 0.000 | 0.611 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.026 | 0.000 | 0.261 |
| linear_ztype__crl_best | full | 0.054 | 0.025 | 0.462 | 0.029 |
| linear_ztype__crl_best | focal | 0.054 | 0.000 | 0.564 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.036 | 0.000 | 0.089 |
| linear_ztype__crl_best_aux_type | full | 0.263 | 0.028 | 0.500 | 0.122 |
| linear_ztype__crl_best_aux_type | focal | 0.269 | 0.000 | 0.611 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.042 | 0.000 | 0.235 |
| mlp_ztype__crl_best | full | 0.084 | 0.058 | 0.479 | 0.061 |
| mlp_ztype__crl_best | focal | 0.084 | 0.000 | 0.590 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.093 | 0.000 | 0.164 |
| mlp_ztype__crl_best_aux_type | full | 0.228 | 0.090 | 0.524 | 0.434 |
| mlp_ztype__crl_best_aux_type | focal | 0.233 | 0.000 | 0.630 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.168 | 0.000 | 0.492 |
