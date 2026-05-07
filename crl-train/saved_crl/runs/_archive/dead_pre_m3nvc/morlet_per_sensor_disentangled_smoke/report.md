# Full diagnostic report — disentangled_smoke

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 4.1 min

- total epochs recorded: 3

- **best val_ref_elbo:** 105.376139 (epoch 0) → `crl_best.pth`

- **best val_aux_type_f1:** 0.1492 (epoch 1) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 0 | 0.6579 | 0.1879 | 0.2428 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 0 | 0.8458 | 0.2002 | 0.3423 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 0 | 0.6568 | 0.2930 | 0.3796 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 1 | 0.6065 | 0.1927 | 0.3174 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.6399 | 0.2730 | 0.2730 | 0.3422 |
| linear_fullz__crl_best | focal | 14,534 | 0.5951 | 0.1890 | 0.1890 | 0.2190 |
| linear_fullz__crl_best | iobt | 936 | 0.5739 | 0.1956 | 0.3912 | 0.3180 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.5949 | 0.1848 | 0.1848 | 0.2972 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.5888 | 0.1256 | 0.1256 | 0.2019 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.5702 | 0.1271 | 0.2542 | 0.3390 |
| linear_signal__crl_best | full | 21,436 | 0.6411 | 0.1608 | 0.1608 | 0.2001 |
| linear_signal__crl_best | focal | 14,534 | 0.5952 | 0.1392 | 0.1392 | 0.1883 |
| linear_signal__crl_best | iobt | 936 | 0.5760 | 0.2090 | 0.4180 | 0.3945 |
| linear_signal__crl_best_aux_type | full | 21,436 | 0.8110 | 0.1900 | 0.1900 | 0.3297 |
| linear_signal__crl_best_aux_type | focal | 14,534 | 0.8103 | 0.1560 | 0.1560 | 0.2440 |
| linear_signal__crl_best_aux_type | iobt | 936 | 0.6944 | 0.1201 | 0.2401 | 0.2295 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.000 | 0.298 | 0.525 | 0.268 |
| linear_fullz__crl_best | focal | 0.000 | 0.287 | 0.214 | 0.255 |
| linear_fullz__crl_best | iobt | 0.000 | 0.459 | 0.000 | 0.324 |
| linear_fullz__crl_best_aux_type | full | 0.000 | 0.000 | 0.427 | 0.312 |
| linear_fullz__crl_best_aux_type | focal | 0.000 | 0.000 | 0.209 | 0.293 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.000 | 0.000 | 0.508 |
| linear_signal__crl_best | full | 0.000 | 0.240 | 0.133 | 0.270 |
| linear_signal__crl_best | focal | 0.000 | 0.286 | 0.013 | 0.258 |
| linear_signal__crl_best | iobt | 0.000 | 0.520 | 0.000 | 0.316 |
| linear_signal__crl_best_aux_type | full | 0.000 | 0.000 | 0.484 | 0.276 |
| linear_signal__crl_best_aux_type | focal | 0.000 | 0.000 | 0.394 | 0.230 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.000 | 0.000 | 0.480 |
