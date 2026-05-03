# Full diagnostic report — 2026-05-03_05-13-25

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 390.66 min

- total epochs recorded: 66

- **best val_ref_elbo:** 802.942783 (epoch 40) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4591 (epoch 6) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 42 | 0.8037 | 0.4718 | 0.6424 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 10 | 0.8303 | 0.4531 | 0.6153 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 9 | 0.4857 | 0.4702 | 0.6546 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 14 | 0.8307 | 0.4525 | 0.6095 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 23 | 0.4860 | 0.3655 | 0.3851 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 9 | 0.8304 | 0.4585 | 0.6253 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.4918 | 0.3485 | 0.3485 | 0.3655 |
| linear_fullz__crl_best | focal | 29,980 | 0.5637 | 0.2804 | 0.2804 | 0.2683 |
| linear_fullz__crl_best | iobt | 1,575 | 0.5527 | 0.1345 | 0.2691 | 0.2551 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8338 | 0.4399 | 0.4399 | 0.5877 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.8046 | 0.3391 | 0.3391 | 0.4286 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.7955 | 0.1291 | 0.2581 | 0.1868 |
| linear_ztype__crl_best | full | 88,631 | 0.8142 | 0.4571 | 0.4571 | 0.6111 |
| linear_ztype__crl_best | focal | 29,980 | 0.8276 | 0.3520 | 0.3520 | 0.4507 |
| linear_ztype__crl_best | iobt | 1,575 | 0.8137 | 0.1787 | 0.3573 | 0.2179 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8336 | 0.4379 | 0.4379 | 0.5827 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.8050 | 0.3369 | 0.3369 | 0.4266 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.7956 | 0.1368 | 0.2737 | 0.2082 |
| mlp_ztype__crl_best | full | 88,631 | 0.4915 | 0.4575 | 0.4575 | 0.6268 |
| mlp_ztype__crl_best | focal | 29,980 | 0.5635 | 0.3310 | 0.3310 | 0.4432 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.5523 | 0.2043 | 0.4086 | 0.2235 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8334 | 0.4386 | 0.4386 | 0.5792 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8071 | 0.3291 | 0.3291 | 0.4164 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.7955 | 0.1390 | 0.2780 | 0.2407 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.379 | 0.264 | 0.395 | 0.356 |
| linear_fullz__crl_best | focal | 0.402 | 0.332 | 0.219 | 0.168 |
| linear_fullz__crl_best | iobt | 0.000 | 0.128 | 0.000 | 0.411 |
| linear_fullz__crl_best_aux_type | full | 0.342 | 0.324 | 0.714 | 0.380 |
| linear_fullz__crl_best_aux_type | focal | 0.379 | 0.337 | 0.535 | 0.105 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.282 | 0.000 | 0.235 |
| linear_ztype__crl_best | full | 0.362 | 0.324 | 0.733 | 0.411 |
| linear_ztype__crl_best | focal | 0.379 | 0.349 | 0.565 | 0.115 |
| linear_ztype__crl_best | iobt | 0.000 | 0.230 | 0.000 | 0.485 |
| linear_ztype__crl_best_aux_type | full | 0.328 | 0.328 | 0.708 | 0.388 |
| linear_ztype__crl_best_aux_type | focal | 0.362 | 0.335 | 0.539 | 0.112 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.321 | 0.000 | 0.226 |
| mlp_ztype__crl_best | full | 0.356 | 0.319 | 0.750 | 0.404 |
| mlp_ztype__crl_best | focal | 0.371 | 0.347 | 0.557 | 0.048 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.263 | 0.000 | 0.554 |
| mlp_ztype__crl_best_aux_type | full | 0.326 | 0.336 | 0.704 | 0.389 |
| mlp_ztype__crl_best_aux_type | focal | 0.363 | 0.341 | 0.520 | 0.092 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.398 | 0.000 | 0.158 |
