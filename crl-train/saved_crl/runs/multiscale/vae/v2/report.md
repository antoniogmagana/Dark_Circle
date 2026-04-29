# Full diagnostic report — multiscale_v2

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 623.63 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.179946 (epoch 93) → `crl_best.pth`

- **best val_aux_type_f1:** 0.5911 (epoch 8) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 42 | 0.8726 | 0.5982 | 0.6998 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 6 | 0.8617 | 0.5924 | 0.6820 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 9 | 0.8669 | 0.5934 | 0.7258 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 3 | 0.8625 | 0.5918 | 0.7003 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 26 | 0.8739 | 0.5986 | 0.7035 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 10 | 0.8611 | 0.5918 | 0.6803 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8731 | 0.5274 | 0.5274 | 0.6674 |
| linear_fullz__crl_best | focal | 32,674 | 0.8582 | 0.4081 | 0.4081 | 0.4109 |
| linear_fullz__crl_best | iobt | 1,575 | 0.7510 | 0.2713 | 0.5426 | 0.3996 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8542 | 0.5508 | 0.5508 | 0.6470 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8484 | 0.4098 | 0.4098 | 0.4145 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.6163 | 0.2538 | 0.5076 | 0.3392 |
| linear_ztype__crl_best | full | 91,325 | 0.8710 | 0.5266 | 0.5266 | 0.6643 |
| linear_ztype__crl_best | focal | 32,674 | 0.8547 | 0.4071 | 0.4071 | 0.4087 |
| linear_ztype__crl_best | iobt | 1,575 | 0.7443 | 0.2647 | 0.5293 | 0.3941 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8545 | 0.5510 | 0.5510 | 0.6476 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8488 | 0.4099 | 0.4099 | 0.4149 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.6176 | 0.2571 | 0.5141 | 0.3467 |
| mlp_ztype__crl_best | full | 91,325 | 0.8639 | 0.5162 | 0.5162 | 0.6863 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8435 | 0.4118 | 0.4118 | 0.4406 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.7158 | 0.2493 | 0.4985 | 0.3067 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8552 | 0.5464 | 0.5464 | 0.6641 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8498 | 0.4123 | 0.4123 | 0.4220 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.6199 | 0.2589 | 0.5179 | 0.3448 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.367 | 0.415 | 0.791 | 0.537 |
| linear_fullz__crl_best | focal | 0.368 | 0.425 | 0.473 | 0.366 |
| linear_fullz__crl_best | iobt | 0.000 | 0.348 | 0.000 | 0.738 |
| linear_fullz__crl_best_aux_type | full | 0.461 | 0.429 | 0.752 | 0.562 |
| linear_fullz__crl_best_aux_type | focal | 0.469 | 0.443 | 0.396 | 0.331 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.303 | 0.000 | 0.712 |
| linear_ztype__crl_best | full | 0.373 | 0.407 | 0.788 | 0.538 |
| linear_ztype__crl_best | focal | 0.374 | 0.418 | 0.463 | 0.372 |
| linear_ztype__crl_best | iobt | 0.000 | 0.336 | 0.000 | 0.723 |
| linear_ztype__crl_best_aux_type | full | 0.462 | 0.427 | 0.752 | 0.563 |
| linear_ztype__crl_best_aux_type | focal | 0.471 | 0.440 | 0.397 | 0.331 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.323 | 0.000 | 0.706 |
| mlp_ztype__crl_best | full | 0.385 | 0.343 | 0.800 | 0.537 |
| mlp_ztype__crl_best | focal | 0.386 | 0.368 | 0.530 | 0.362 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.176 | 0.000 | 0.821 |
| mlp_ztype__crl_best_aux_type | full | 0.436 | 0.415 | 0.770 | 0.564 |
| mlp_ztype__crl_best_aux_type | focal | 0.442 | 0.429 | 0.452 | 0.326 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.296 | 0.000 | 0.740 |
