# Full diagnostic report — 2026-04-28_23-12-59

## CRL pre-training

- frontend: `multiscale`, d_z=32, d_model=64, n_layers=2

- elapsed: 590.12 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.21401 (epoch 95) → `crl_best.pth`

- **best val_aux_type_f1:** 0.648 (epoch 24) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 0 | 0.0000 | 0.4882 | 0.7282 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 44 | 0.8731 | 0.6377 | 0.7430 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 26 | 0.8734 | 0.6041 | 0.7411 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 1 | 0.8731 | 0.6306 | 0.7626 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 0 | 0.0000 | 0.5220 | 0.7326 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 22 | 0.8712 | 0.6385 | 0.7425 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.0000 | 0.4853 | 0.4853 | 0.7076 |
| linear_fullz__crl_best | focal | 32,674 | 0.0000 | 0.3491 | 0.3491 | 0.4115 |
| linear_fullz__crl_best | iobt | 1,575 | 0.0000 | 0.3225 | 0.6449 | 0.4721 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8686 | 0.5874 | 0.5874 | 0.7163 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8403 | 0.4483 | 0.4483 | 0.4484 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.6822 | 0.3040 | 0.6081 | 0.4870 |
| linear_ztype__crl_best | full | 91,325 | 0.0000 | 0.4667 | 0.4667 | 0.7114 |
| linear_ztype__crl_best | focal | 32,674 | 0.0000 | 0.3359 | 0.3359 | 0.4203 |
| linear_ztype__crl_best | iobt | 1,575 | 0.0000 | 0.3180 | 0.6361 | 0.4582 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8712 | 0.5864 | 0.5864 | 0.7166 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8451 | 0.4467 | 0.4467 | 0.4476 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.6940 | 0.3082 | 0.6164 | 0.4926 |
| mlp_ztype__crl_best | full | 91,325 | 0.8733 | 0.5418 | 0.5418 | 0.7053 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8555 | 0.4105 | 0.4105 | 0.4258 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.7289 | 0.2738 | 0.5476 | 0.4257 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8712 | 0.5830 | 0.5830 | 0.7362 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8454 | 0.4587 | 0.4587 | 0.4792 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.6952 | 0.2916 | 0.5832 | 0.3950 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.053 | 0.461 | 0.829 | 0.598 |
| linear_fullz__crl_best | focal | 0.053 | 0.461 | 0.573 | 0.309 |
| linear_fullz__crl_best | iobt | 0.000 | 0.472 | 0.000 | 0.818 |
| linear_fullz__crl_best_aux_type | full | 0.419 | 0.463 | 0.825 | 0.643 |
| linear_fullz__crl_best_aux_type | focal | 0.422 | 0.459 | 0.478 | 0.435 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.499 | 0.000 | 0.717 |
| linear_ztype__crl_best | full | 0.000 | 0.424 | 0.830 | 0.613 |
| linear_ztype__crl_best | focal | 0.000 | 0.421 | 0.588 | 0.334 |
| linear_ztype__crl_best | iobt | 0.000 | 0.448 | 0.000 | 0.824 |
| linear_ztype__crl_best_aux_type | full | 0.419 | 0.455 | 0.825 | 0.646 |
| linear_ztype__crl_best_aux_type | focal | 0.422 | 0.451 | 0.478 | 0.436 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.509 | 0.000 | 0.724 |
| mlp_ztype__crl_best | full | 0.306 | 0.433 | 0.820 | 0.608 |
| mlp_ztype__crl_best | focal | 0.308 | 0.441 | 0.528 | 0.364 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.382 | 0.000 | 0.714 |
| mlp_ztype__crl_best_aux_type | full | 0.415 | 0.434 | 0.834 | 0.649 |
| mlp_ztype__crl_best_aux_type | focal | 0.417 | 0.447 | 0.558 | 0.412 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.350 | 0.000 | 0.816 |
