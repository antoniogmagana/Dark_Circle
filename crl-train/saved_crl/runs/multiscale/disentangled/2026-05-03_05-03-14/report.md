# Full diagnostic report — 2026-05-03_05-03-14

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 549.28 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.252379 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6949 (epoch 32) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 29 | 0.8369 | 0.6403 | 0.7660 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 16 | 0.7722 | 0.6919 | 0.7797 |
| mlp_signal__crl_best | mlp_signal | crl_best.pth | 48 | 0.8130 | 0.6655 | 0.7694 |
| mlp_signal__crl_best_aux_type | mlp_signal | crl_best_aux_type.pth | 21 | 0.7656 | 0.6917 | 0.7764 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 36 | 0.7822 | 0.6460 | 0.7661 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 17 | 0.7179 | 0.6901 | 0.7751 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.7496 | 0.5844 | 0.5844 | 0.7318 |
| linear_fullz__crl_best | focal | 29,980 | 0.4958 | 0.4245 | 0.4245 | 0.4792 |
| linear_fullz__crl_best | iobt | 1,575 | 0.4238 | 0.2888 | 0.5776 | 0.3727 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.6848 | 0.6266 | 0.6266 | 0.7272 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.1290 | 0.4351 | 0.4351 | 0.4796 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.3326 | 0.3239 | 0.6478 | 0.5000 |
| linear_signal__crl_best | full | 88,631 | 0.8245 | 0.5789 | 0.5789 | 0.7316 |
| linear_signal__crl_best | focal | 29,980 | 0.6733 | 0.4215 | 0.4215 | 0.4796 |
| linear_signal__crl_best | iobt | 1,575 | 0.7085 | 0.2826 | 0.5652 | 0.3532 |
| linear_signal__crl_best_aux_type | full | 88,631 | 0.7472 | 0.6303 | 0.6303 | 0.7330 |
| linear_signal__crl_best_aux_type | focal | 29,980 | 0.2658 | 0.4293 | 0.4293 | 0.4788 |
| linear_signal__crl_best_aux_type | iobt | 1,575 | 0.4273 | 0.3363 | 0.6726 | 0.5093 |
| mlp_signal__crl_best | full | 88,631 | 0.7842 | 0.6021 | 0.6021 | 0.7291 |
| mlp_signal__crl_best | focal | 29,980 | 0.5665 | 0.4306 | 0.4306 | 0.4727 |
| mlp_signal__crl_best | iobt | 1,575 | 0.5524 | 0.3097 | 0.6195 | 0.4238 |
| mlp_signal__crl_best_aux_type | full | 88,631 | 0.7391 | 0.6284 | 0.6284 | 0.7288 |
| mlp_signal__crl_best_aux_type | focal | 29,980 | 0.2160 | 0.4374 | 0.4374 | 0.4773 |
| mlp_signal__crl_best_aux_type | iobt | 1,575 | 0.4206 | 0.3197 | 0.6394 | 0.4777 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.406 | 0.465 | 0.827 | 0.639 |
| linear_fullz__crl_best | focal | 0.407 | 0.484 | 0.579 | 0.229 |
| linear_fullz__crl_best | iobt | 0.000 | 0.315 | 0.000 | 0.840 |
| linear_fullz__crl_best_aux_type | full | 0.502 | 0.519 | 0.821 | 0.664 |
| linear_fullz__crl_best_aux_type | focal | 0.504 | 0.520 | 0.506 | 0.210 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.510 | 0.000 | 0.786 |
| linear_signal__crl_best | full | 0.404 | 0.448 | 0.826 | 0.639 |
| linear_signal__crl_best | focal | 0.404 | 0.470 | 0.578 | 0.234 |
| linear_signal__crl_best | iobt | 0.000 | 0.279 | 0.000 | 0.851 |
| linear_signal__crl_best_aux_type | full | 0.503 | 0.519 | 0.826 | 0.672 |
| linear_signal__crl_best_aux_type | focal | 0.505 | 0.519 | 0.499 | 0.194 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.523 | 0.000 | 0.822 |
| mlp_signal__crl_best | full | 0.456 | 0.485 | 0.824 | 0.643 |
| mlp_signal__crl_best | focal | 0.457 | 0.496 | 0.542 | 0.227 |
| mlp_signal__crl_best | iobt | 0.000 | 0.401 | 0.000 | 0.838 |
| mlp_signal__crl_best_aux_type | full | 0.509 | 0.515 | 0.821 | 0.669 |
| mlp_signal__crl_best_aux_type | focal | 0.511 | 0.519 | 0.486 | 0.233 |
| mlp_signal__crl_best_aux_type | iobt | 0.000 | 0.479 | 0.000 | 0.799 |
