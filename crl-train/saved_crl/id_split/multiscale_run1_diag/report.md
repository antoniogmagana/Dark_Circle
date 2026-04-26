# Full diagnostic report — multiscale_run1_diag

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- total epochs recorded: 100

- **best val_ref_elbo:** 0.170033 (epoch 97) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4802 (epoch 27) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 15 | 0.8465 | 0.4701 | 0.4754 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 46 | 0.8416 | 0.5010 | 0.5022 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 26 | 0.8480 | 0.4618 | 0.4648 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 25 | 0.8416 | 0.5001 | 0.5062 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 17 | 0.8480 | 0.4718 | 0.4770 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 49 | 0.8436 | 0.4979 | 0.4995 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 17,123 | 0.7965 | 0.4500 | 0.4500 | 0.4589 |
| linear_fullz__crl_best | focal | 16,336 | 0.7972 | 0.4471 | 0.4471 | 0.4587 |
| linear_fullz__crl_best | iobt | 787 | 0.7811 | 0.2621 | 0.5243 | 0.4632 |
| linear_fullz__crl_best_aux_type | full | 17,123 | 0.7957 | 0.4438 | 0.4438 | 0.4398 |
| linear_fullz__crl_best_aux_type | focal | 16,336 | 0.7987 | 0.4403 | 0.4403 | 0.4357 |
| linear_fullz__crl_best_aux_type | iobt | 787 | 0.7308 | 0.2816 | 0.5632 | 0.5239 |
| linear_ztype__crl_best | full | 17,123 | 0.7950 | 0.4422 | 0.4422 | 0.4526 |
| linear_ztype__crl_best | focal | 16,336 | 0.7957 | 0.4380 | 0.4380 | 0.4513 |
| linear_ztype__crl_best | iobt | 787 | 0.7799 | 0.2685 | 0.5370 | 0.4779 |
| linear_ztype__crl_best_aux_type | full | 17,123 | 0.7943 | 0.4435 | 0.4435 | 0.4403 |
| linear_ztype__crl_best_aux_type | focal | 16,336 | 0.7977 | 0.4393 | 0.4393 | 0.4359 |
| linear_ztype__crl_best_aux_type | iobt | 787 | 0.7210 | 0.2851 | 0.5701 | 0.5312 |
| mlp_ztype__crl_best | full | 17,123 | 0.7964 | 0.4393 | 0.4393 | 0.4461 |
| mlp_ztype__crl_best | focal | 16,336 | 0.7971 | 0.4375 | 0.4375 | 0.4460 |
| mlp_ztype__crl_best | iobt | 787 | 0.7823 | 0.2468 | 0.4936 | 0.4485 |
| mlp_ztype__crl_best_aux_type | full | 17,123 | 0.7945 | 0.4536 | 0.4536 | 0.4516 |
| mlp_ztype__crl_best_aux_type | focal | 16,336 | 0.7978 | 0.4522 | 0.4522 | 0.4501 |
| mlp_ztype__crl_best_aux_type | iobt | 787 | 0.7223 | 0.2612 | 0.5223 | 0.4835 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.438 | 0.475 | 0.499 | 0.387 |
| linear_fullz__crl_best | focal | 0.442 | 0.475 | 0.503 | 0.368 |
| linear_fullz__crl_best | iobt | 0.000 | 0.478 | 0.000 | 0.571 |
| linear_fullz__crl_best_aux_type | full | 0.424 | 0.512 | 0.410 | 0.429 |
| linear_fullz__crl_best_aux_type | focal | 0.428 | 0.509 | 0.411 | 0.413 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.535 | 0.000 | 0.591 |
| linear_ztype__crl_best | full | 0.424 | 0.457 | 0.501 | 0.386 |
| linear_ztype__crl_best | focal | 0.428 | 0.453 | 0.505 | 0.366 |
| linear_ztype__crl_best | iobt | 0.000 | 0.496 | 0.000 | 0.578 |
| linear_ztype__crl_best_aux_type | full | 0.424 | 0.505 | 0.413 | 0.432 |
| linear_ztype__crl_best_aux_type | focal | 0.428 | 0.499 | 0.414 | 0.416 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.549 | 0.000 | 0.592 |
| mlp_ztype__crl_best | full | 0.425 | 0.454 | 0.490 | 0.388 |
| mlp_ztype__crl_best | focal | 0.428 | 0.457 | 0.494 | 0.371 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.432 | 0.000 | 0.555 |
| mlp_ztype__crl_best_aux_type | full | 0.428 | 0.505 | 0.452 | 0.429 |
| mlp_ztype__crl_best_aux_type | focal | 0.432 | 0.509 | 0.453 | 0.414 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.476 | 0.000 | 0.569 |
