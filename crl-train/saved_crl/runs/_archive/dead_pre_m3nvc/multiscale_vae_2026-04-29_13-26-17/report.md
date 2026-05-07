# Full diagnostic report — 2026-04-29_13-26-17

## CRL pre-training

- frontend: `multiscale`, d_z=32, d_model=64, n_layers=2

- elapsed: 512.68 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.192563 (epoch 98) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6768 (epoch 35) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 5 | 0.8730 | 0.6505 | 0.7659 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 48 | 0.8675 | 0.6756 | 0.7762 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 2 | 0.8705 | 0.6619 | 0.7844 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 1 | 0.8731 | 0.6644 | 0.7890 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 26 | 0.8696 | 0.6479 | 0.7622 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 11 | 0.8683 | 0.6765 | 0.7779 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.8658 | 0.5764 | 0.5764 | 0.7172 |
| linear_fullz__crl_best | focal | 29,980 | 0.8450 | 0.3994 | 0.3994 | 0.4429 |
| linear_fullz__crl_best | iobt | 1,575 | 0.7082 | 0.2617 | 0.5233 | 0.3708 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8611 | 0.6011 | 0.6011 | 0.7230 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.8240 | 0.4152 | 0.4152 | 0.4514 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.6831 | 0.3066 | 0.6132 | 0.4963 |
| linear_ztype__crl_best | full | 88,631 | 0.8705 | 0.5791 | 0.5791 | 0.7213 |
| linear_ztype__crl_best | focal | 29,980 | 0.8515 | 0.4018 | 0.4018 | 0.4484 |
| linear_ztype__crl_best | iobt | 1,575 | 0.7284 | 0.2667 | 0.5334 | 0.3708 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8601 | 0.6002 | 0.6002 | 0.7213 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.8223 | 0.4132 | 0.4132 | 0.4483 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.6759 | 0.3106 | 0.6211 | 0.5084 |
| mlp_ztype__crl_best | full | 88,631 | 0.8673 | 0.5914 | 0.5914 | 0.7419 |
| mlp_ztype__crl_best | focal | 29,980 | 0.8474 | 0.4068 | 0.4068 | 0.4767 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.7174 | 0.2981 | 0.5963 | 0.3736 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8668 | 0.5956 | 0.5956 | 0.7454 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8332 | 0.4194 | 0.4194 | 0.4874 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.7019 | 0.2800 | 0.5600 | 0.3745 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.409 | 0.442 | 0.817 | 0.637 |
| linear_fullz__crl_best | focal | 0.411 | 0.462 | 0.546 | 0.178 |
| linear_fullz__crl_best | iobt | 0.000 | 0.291 | 0.000 | 0.756 |
| linear_fullz__crl_best_aux_type | full | 0.450 | 0.477 | 0.821 | 0.656 |
| linear_fullz__crl_best_aux_type | focal | 0.455 | 0.476 | 0.519 | 0.211 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.499 | 0.000 | 0.728 |
| linear_ztype__crl_best | full | 0.408 | 0.444 | 0.819 | 0.645 |
| linear_ztype__crl_best | focal | 0.410 | 0.464 | 0.550 | 0.183 |
| linear_ztype__crl_best | iobt | 0.000 | 0.291 | 0.000 | 0.776 |
| linear_ztype__crl_best_aux_type | full | 0.448 | 0.478 | 0.821 | 0.653 |
| linear_ztype__crl_best_aux_type | focal | 0.453 | 0.476 | 0.516 | 0.208 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.516 | 0.000 | 0.726 |
| mlp_ztype__crl_best | full | 0.407 | 0.442 | 0.830 | 0.686 |
| mlp_ztype__crl_best | focal | 0.409 | 0.459 | 0.571 | 0.189 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.315 | 0.000 | 0.878 |
| mlp_ztype__crl_best_aux_type | full | 0.434 | 0.433 | 0.832 | 0.683 |
| mlp_ztype__crl_best_aux_type | focal | 0.438 | 0.452 | 0.580 | 0.208 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.299 | 0.000 | 0.821 |
