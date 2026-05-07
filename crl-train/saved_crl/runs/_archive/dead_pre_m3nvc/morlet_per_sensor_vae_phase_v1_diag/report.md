# Full diagnostic report — morlet_per_sensor_phase_run1_diag

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 490.18 min

- total epochs recorded: 78

- **best val_ref_elbo:** 2.276932 (epoch 52) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4212 (epoch 39) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 14 | 0.7263 | 0.4550 | 0.5937 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 35 | 0.7787 | 0.5152 | 0.6318 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 7 | 0.7706 | 0.4297 | 0.5944 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 2 | 0.7720 | 0.5151 | 0.6383 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 48 | 0.7800 | 0.4515 | 0.5828 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 8 | 0.7779 | 0.4683 | 0.5801 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.7816 | 0.3969 | 0.3969 | 0.5371 |
| linear_fullz__crl_best | focal | 32,674 | 0.8230 | 0.2902 | 0.2902 | 0.3042 |
| linear_fullz__crl_best | iobt | 1,575 | 0.8007 | 0.1812 | 0.3623 | 0.4164 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.7791 | 0.4139 | 0.4139 | 0.5300 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8251 | 0.2939 | 0.2939 | 0.3065 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.7942 | 0.1628 | 0.3256 | 0.3862 |
| linear_ztype__crl_best | full | 91,325 | 0.7228 | 0.4446 | 0.4446 | 0.5827 |
| linear_ztype__crl_best | focal | 32,674 | 0.7761 | 0.3568 | 0.3568 | 0.3589 |
| linear_ztype__crl_best | iobt | 1,575 | 0.7357 | 0.2562 | 0.5124 | 0.4800 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.7796 | 0.4840 | 0.4840 | 0.6079 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8245 | 0.3725 | 0.3725 | 0.3874 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.7930 | 0.2680 | 0.5359 | 0.4916 |
| mlp_ztype__crl_best | full | 91,325 | 0.7711 | 0.4115 | 0.4115 | 0.5748 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8182 | 0.3074 | 0.3074 | 0.3496 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.7913 | 0.2693 | 0.5386 | 0.4823 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.7765 | 0.4782 | 0.4782 | 0.6100 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8374 | 0.3634 | 0.3634 | 0.3690 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.8164 | 0.2423 | 0.4845 | 0.4577 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.310 | 0.278 | 0.725 | 0.275 |
| linear_fullz__crl_best | focal | 0.314 | 0.324 | 0.330 | 0.193 |
| linear_fullz__crl_best | iobt | 0.000 | 0.568 | 0.000 | 0.156 |
| linear_fullz__crl_best_aux_type | full | 0.367 | 0.267 | 0.710 | 0.311 |
| linear_fullz__crl_best_aux_type | focal | 0.376 | 0.314 | 0.297 | 0.189 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.542 | 0.000 | 0.109 |
| linear_ztype__crl_best | full | 0.365 | 0.274 | 0.748 | 0.392 |
| linear_ztype__crl_best | focal | 0.371 | 0.297 | 0.436 | 0.323 |
| linear_ztype__crl_best | iobt | 0.000 | 0.593 | 0.000 | 0.431 |
| linear_ztype__crl_best_aux_type | full | 0.426 | 0.348 | 0.761 | 0.401 |
| linear_ztype__crl_best_aux_type | focal | 0.442 | 0.377 | 0.416 | 0.254 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.586 | 0.000 | 0.486 |
| mlp_ztype__crl_best | full | 0.284 | 0.277 | 0.749 | 0.336 |
| mlp_ztype__crl_best | focal | 0.286 | 0.324 | 0.475 | 0.144 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.604 | 0.000 | 0.473 |
| mlp_ztype__crl_best_aux_type | full | 0.404 | 0.275 | 0.760 | 0.473 |
| mlp_ztype__crl_best_aux_type | focal | 0.409 | 0.309 | 0.425 | 0.311 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.556 | 0.000 | 0.413 |
