# Full diagnostic report — 2026-04-22_02-53-55

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 138.55 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.274601 (epoch 98) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6831 (epoch 25) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 25 | 0.8572 | 0.6692 | 0.7243 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 27 | 0.8642 | 0.6956 | 0.7538 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 4 | 0.8638 | 0.6593 | 0.7497 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 24 | 0.8643 | 0.7010 | 0.7584 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 23 | 0.8595 | 0.6685 | 0.7210 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 7 | 0.8635 | 0.6958 | 0.7546 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_f1 | type_f1_support_only | pres_f1_cal | type_f1_cal | type_f1_support_only_cal | type_acc |
|---|---|---|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7542 | 0.2549 | 0.2549 | 0.8133 | 0.1380 | 0.1380 | 0.2594 |
| linear_fullz__crl_best | focal | 9,228 | 0.7585 | 0.1313 | 0.2626 | 0.8259 | 0.1629 | 0.3258 | 0.2018 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7420 | 0.2546 | 0.5091 | 0.7755 | 0.2059 | 0.4117 | 0.4436 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.7689 | 0.2468 | 0.2468 | 0.8133 | 0.1487 | 0.1487 | 0.2549 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.7849 | 0.1372 | 0.2743 | 0.8259 | 0.1729 | 0.3458 | 0.2132 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.7185 | 0.2102 | 0.4203 | 0.7755 | 0.2059 | 0.4117 | 0.3884 |
| linear_ztype__crl_best | full | 12,432 | 0.7482 | 0.2529 | 0.2529 | 0.8133 | 0.1374 | 0.1374 | 0.2606 |
| linear_ztype__crl_best | focal | 9,228 | 0.7515 | 0.1343 | 0.2687 | 0.8259 | 0.1623 | 0.3246 | 0.2088 |
| linear_ztype__crl_best | iobt | 3,204 | 0.7389 | 0.2449 | 0.4898 | 0.7755 | 0.2059 | 0.4117 | 0.4263 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7693 | 0.2583 | 0.2583 | 0.8133 | 0.1474 | 0.1474 | 0.2579 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7854 | 0.1320 | 0.2640 | 0.8259 | 0.1715 | 0.3431 | 0.2024 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.7190 | 0.2408 | 0.4815 | 0.7755 | 0.2067 | 0.4135 | 0.4352 |
| mlp_ztype__crl_best | full | 12,432 | 0.7614 | 0.2545 | 0.2545 | 0.8133 | 0.1351 | 0.1351 | 0.3370 |
| mlp_ztype__crl_best | focal | 9,228 | 0.7663 | 0.1752 | 0.3504 | 0.8259 | 0.1608 | 0.3216 | 0.3574 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7475 | 0.1728 | 0.3456 | 0.7755 | 0.2059 | 0.4117 | 0.2716 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.7697 | 0.2713 | 0.2713 | 0.8133 | 0.1447 | 0.1447 | 0.2637 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.7857 | 0.1286 | 0.2572 | 0.8259 | 0.1691 | 0.3382 | 0.1983 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.7194 | 0.2695 | 0.5390 | 0.7755 | 0.2414 | 0.4829 | 0.4731 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.112 | 0.217 | 0.383 | 0.307 |
| linear_fullz__crl_best | focal | 0.116 | 0.000 | 0.409 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.392 | 0.000 | 0.626 |
| linear_fullz__crl_best_aux_type | full | 0.158 | 0.163 | 0.368 | 0.298 |
| linear_fullz__crl_best_aux_type | focal | 0.165 | 0.000 | 0.384 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.254 | 0.000 | 0.586 |
| linear_ztype__crl_best | full | 0.112 | 0.205 | 0.393 | 0.301 |
| linear_ztype__crl_best | focal | 0.116 | 0.000 | 0.421 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.359 | 0.000 | 0.620 |
| linear_ztype__crl_best_aux_type | full | 0.153 | 0.191 | 0.352 | 0.337 |
| linear_ztype__crl_best_aux_type | focal | 0.160 | 0.000 | 0.368 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.351 | 0.000 | 0.612 |
| mlp_ztype__crl_best | full | 0.099 | 0.048 | 0.533 | 0.337 |
| mlp_ztype__crl_best | focal | 0.102 | 0.000 | 0.598 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.052 | 0.000 | 0.639 |
| mlp_ztype__crl_best_aux_type | full | 0.141 | 0.217 | 0.347 | 0.379 |
| mlp_ztype__crl_best_aux_type | focal | 0.147 | 0.000 | 0.368 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.441 | 0.000 | 0.637 |
