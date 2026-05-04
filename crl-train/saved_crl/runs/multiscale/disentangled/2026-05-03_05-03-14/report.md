# Full diagnostic report — 2026-05-03_05-03-14

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 549.28 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.252379 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6949 (epoch 32) → `crl_best_aux_type.pth`


## Phase 2 — probes (selected by max val F1, per head)

Each probe trains both heads jointly with two independent optimizers and saves two checkpoints: the presence ckpt is the epoch with max `val_pres_f1`, the type ckpt is the epoch with max `val_type_f1`. These epochs may differ.

### Presence head

| run | probe | ckpt | best_epoch | val_pres_f1 | val_pres_acc |
|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 44 | 0.8438 | 0.7492 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 30 | 0.7944 | 0.7011 |
| mlp_signal__crl_best | mlp_signal | crl_best.pth | 30 | 0.8418 | 0.7503 |
| mlp_signal__crl_best_aux_type | mlp_signal | crl_best_aux_type.pth | 28 | 0.7946 | 0.6875 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 3 | 0.8433 | 0.7558 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 39 | 0.7931 | 0.6995 |

### Type head

| run | probe | ckpt | best_epoch | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 0 | 0.6503 | 0.7557 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 40 | 0.6702 | 0.7603 |
| mlp_signal__crl_best | mlp_signal | crl_best.pth | 2 | 0.6535 | 0.7503 |
| mlp_signal__crl_best_aux_type | mlp_signal | crl_best_aux_type.pth | 46 | 0.6729 | 0.7700 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 1 | 0.6462 | 0.7499 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 22 | 0.6704 | 0.7588 |

## Phase 3 — test evals

Each eval row is from a single head's checkpoint: presence rows come from `downstream_best_pres.pth`, type rows from `downstream_best_type.pth`. Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt, m3nvc, per-vehicle) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

### Presence head — test pres_f1 by split

| run | split | n_windows | pres_f1 |
|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8353 |
| linear_fullz__crl_best | focal | 32,674 | 0.7312 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.7756 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.9158 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.6910 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.5758 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.8981 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.8850 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.8311 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.4530 |
| linear_fullz__crl_best | iobt | 1,575 | 0.7780 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.7706 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.7925 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.7867 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.8800 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.8741 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.8801 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.8716 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.8937 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.7753 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.4644 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.3445 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.7668 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.2698 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.3273 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.7308 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.7226 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.5275 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.0946 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.5621 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.4442 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7805 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.5702 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8799 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8757 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8784 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8703 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8947 |
| linear_signal__crl_best | full | 91,325 | 0.8463 |
| linear_signal__crl_best | focal | 32,674 | 0.7891 |
| linear_signal__crl_best | focal__bicycle2 | 3,911 | 0.8081 |
| linear_signal__crl_best | focal__forester2 | 2,975 | 0.9205 |
| linear_signal__crl_best | focal__motor2 | 2,734 | 0.6550 |
| linear_signal__crl_best | focal__mustang0528 | 10,643 | 0.7058 |
| linear_signal__crl_best | focal__pickup2 | 2,548 | 0.9055 |
| linear_signal__crl_best | focal__scooter2 | 2,762 | 0.8857 |
| linear_signal__crl_best | focal__tesla2 | 2,694 | 0.8669 |
| linear_signal__crl_best | focal__walk2 | 4,407 | 0.6734 |
| linear_signal__crl_best | iobt | 1,575 | 0.7860 |
| linear_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.7771 |
| linear_signal__crl_best | iobt__silverado0315pm | 490 | 0.7845 |
| linear_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.8287 |
| linear_signal__crl_best | m3nvc | 57,076 | 0.8729 |
| linear_signal__crl_best | m3nvc__cx30 | 14,020 | 0.8635 |
| linear_signal__crl_best | m3nvc__gle350 | 15,317 | 0.8757 |
| linear_signal__crl_best | m3nvc__miata | 13,441 | 0.8669 |
| linear_signal__crl_best | m3nvc__mustang | 14,298 | 0.8845 |
| linear_signal__crl_best_aux_type | full | 91,325 | 0.7760 |
| linear_signal__crl_best_aux_type | focal | 32,674 | 0.4659 |
| linear_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.3678 |
| linear_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.7657 |
| linear_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.2988 |
| linear_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2959 |
| linear_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.7431 |
| linear_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.7224 |
| linear_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.5455 |
| linear_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.0946 |
| linear_signal__crl_best_aux_type | iobt | 1,575 | 0.5740 |
| linear_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.4617 |
| linear_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7795 |
| linear_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.5579 |
| linear_signal__crl_best_aux_type | m3nvc | 57,076 | 0.8809 |
| linear_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8777 |
| linear_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8783 |
| linear_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8702 |
| linear_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8968 |
| mlp_signal__crl_best | full | 91,325 | 0.8401 |
| mlp_signal__crl_best | focal | 32,674 | 0.7601 |
| mlp_signal__crl_best | focal__bicycle2 | 3,911 | 0.7871 |
| mlp_signal__crl_best | focal__forester2 | 2,975 | 0.9189 |
| mlp_signal__crl_best | focal__motor2 | 2,734 | 0.5896 |
| mlp_signal__crl_best | focal__mustang0528 | 10,643 | 0.6619 |
| mlp_signal__crl_best | focal__pickup2 | 2,548 | 0.8974 |
| mlp_signal__crl_best | focal__scooter2 | 2,762 | 0.8835 |
| mlp_signal__crl_best | focal__tesla2 | 2,694 | 0.8475 |
| mlp_signal__crl_best | focal__walk2 | 4,407 | 0.5988 |
| mlp_signal__crl_best | iobt | 1,575 | 0.7665 |
| mlp_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.7440 |
| mlp_signal__crl_best | iobt__silverado0315pm | 490 | 0.7919 |
| mlp_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.8220 |
| mlp_signal__crl_best | m3nvc | 57,076 | 0.8760 |
| mlp_signal__crl_best | m3nvc__cx30 | 14,020 | 0.8687 |
| mlp_signal__crl_best | m3nvc__gle350 | 15,317 | 0.8781 |
| mlp_signal__crl_best | m3nvc__miata | 13,441 | 0.8677 |
| mlp_signal__crl_best | m3nvc__mustang | 14,298 | 0.8888 |
| mlp_signal__crl_best_aux_type | full | 91,325 | 0.7836 |
| mlp_signal__crl_best_aux_type | focal | 32,674 | 0.5415 |
| mlp_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.4523 |
| mlp_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.8315 |
| mlp_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.3059 |
| mlp_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.4307 |
| mlp_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.7802 |
| mlp_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.7849 |
| mlp_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.6153 |
| mlp_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.1383 |
| mlp_signal__crl_best_aux_type | iobt | 1,575 | 0.6201 |
| mlp_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.5405 |
| mlp_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7576 |
| mlp_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6385 |
| mlp_signal__crl_best_aux_type | m3nvc | 57,076 | 0.8686 |
| mlp_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8575 |
| mlp_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8724 |
| mlp_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8638 |
| mlp_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8797 |

### Type head — test type macro_f1 by split

| run | split | n_windows | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.5831 | 0.5831 | 0.7125 |
| linear_fullz__crl_best | focal | 32,674 | 0.4354 | 0.4354 | 0.4552 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.0438 | 0.1753 | 0.0961 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.1897 | 0.7588 | 0.6113 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.2034 | 0.8137 | 0.6859 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.1740 | 0.6959 | 0.5336 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.1221 | 0.4883 | 0.3230 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.0887 | 0.3550 | 0.2158 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.1136 | 0.4544 | 0.2940 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.1959 | 0.7837 | 0.6444 |
| linear_fullz__crl_best | iobt | 1,575 | 0.2932 | 0.5864 | 0.4071 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1017 | 0.4067 | 0.2553 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.2434 | 0.9734 | 0.9481 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.0414 | 0.1656 | 0.0903 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.4097 | 0.8195 | 0.8582 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.2344 | 0.9378 | 0.8829 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.2107 | 0.8429 | 0.7285 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.2365 | 0.9459 | 0.8974 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.2418 | 0.9672 | 0.9365 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.6060 | 0.6060 | 0.7102 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.4335 | 0.4335 | 0.4458 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0849 | 0.3397 | 0.2046 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.1183 | 0.4734 | 0.3101 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.2215 | 0.8860 | 0.7953 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1334 | 0.5337 | 0.3640 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.1133 | 0.4531 | 0.2929 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1698 | 0.6792 | 0.5143 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1150 | 0.4600 | 0.2987 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.2202 | 0.8808 | 0.7869 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.3065 | 0.6129 | 0.5019 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1410 | 0.5640 | 0.3927 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2448 | 0.9792 | 0.9593 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0636 | 0.2545 | 0.1458 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.4109 | 0.8218 | 0.8574 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2328 | 0.9311 | 0.8711 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2154 | 0.8617 | 0.7571 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2367 | 0.9468 | 0.8989 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2386 | 0.9544 | 0.9128 |
| linear_signal__crl_best | full | 91,325 | 0.5814 | 0.5814 | 0.7065 |
| linear_signal__crl_best | focal | 32,674 | 0.4112 | 0.4112 | 0.4339 |
| linear_signal__crl_best | focal__bicycle2 | 3,911 | 0.0502 | 0.2008 | 0.1116 |
| linear_signal__crl_best | focal__forester2 | 2,975 | 0.1828 | 0.7311 | 0.5762 |
| linear_signal__crl_best | focal__motor2 | 2,734 | 0.2195 | 0.8778 | 0.7822 |
| linear_signal__crl_best | focal__mustang0528 | 10,643 | 0.1488 | 0.5953 | 0.4238 |
| linear_signal__crl_best | focal__pickup2 | 2,548 | 0.0916 | 0.3664 | 0.2243 |
| linear_signal__crl_best | focal__scooter2 | 2,762 | 0.1256 | 0.5025 | 0.3356 |
| linear_signal__crl_best | focal__tesla2 | 2,694 | 0.0835 | 0.3341 | 0.2006 |
| linear_signal__crl_best | focal__walk2 | 4,407 | 0.2053 | 0.8212 | 0.6967 |
| linear_signal__crl_best | iobt | 1,575 | 0.3327 | 0.6655 | 0.4842 |
| linear_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1351 | 0.5402 | 0.3701 |
| linear_signal__crl_best | iobt__silverado0315pm | 490 | 0.2434 | 0.9734 | 0.9481 |
| linear_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.0610 | 0.2439 | 0.1389 |
| linear_signal__crl_best | m3nvc | 57,076 | 0.4045 | 0.8091 | 0.8585 |
| linear_signal__crl_best | m3nvc__cx30 | 14,020 | 0.2402 | 0.9609 | 0.9247 |
| linear_signal__crl_best | m3nvc__gle350 | 15,317 | 0.1958 | 0.7833 | 0.6438 |
| linear_signal__crl_best | m3nvc__miata | 13,441 | 0.2415 | 0.9659 | 0.9341 |
| linear_signal__crl_best | m3nvc__mustang | 14,298 | 0.2442 | 0.9767 | 0.9544 |
| linear_signal__crl_best_aux_type | full | 91,325 | 0.6069 | 0.6069 | 0.7138 |
| linear_signal__crl_best_aux_type | focal | 32,674 | 0.4365 | 0.4365 | 0.4482 |
| linear_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0840 | 0.3359 | 0.2019 |
| linear_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.1216 | 0.4864 | 0.3214 |
| linear_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.2209 | 0.8835 | 0.7912 |
| linear_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1351 | 0.5405 | 0.3704 |
| linear_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.1165 | 0.4659 | 0.3037 |
| linear_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1667 | 0.6667 | 0.5000 |
| linear_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1179 | 0.4717 | 0.3086 |
| linear_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.2197 | 0.8786 | 0.7835 |
| linear_signal__crl_best_aux_type | iobt | 1,575 | 0.3073 | 0.6147 | 0.5009 |
| linear_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1410 | 0.5640 | 0.3927 |
| linear_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2448 | 0.9792 | 0.9593 |
| linear_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0610 | 0.2439 | 0.1389 |
| linear_signal__crl_best_aux_type | m3nvc | 57,076 | 0.4124 | 0.8248 | 0.8618 |
| linear_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2349 | 0.9398 | 0.8865 |
| linear_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2130 | 0.8520 | 0.7422 |
| linear_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2380 | 0.9520 | 0.9084 |
| linear_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2399 | 0.9598 | 0.9226 |
| mlp_signal__crl_best | full | 91,325 | 0.5864 | 0.5864 | 0.6980 |
| mlp_signal__crl_best | focal | 32,674 | 0.4014 | 0.4014 | 0.4204 |
| mlp_signal__crl_best | focal__bicycle2 | 3,911 | 0.0765 | 0.3059 | 0.1806 |
| mlp_signal__crl_best | focal__forester2 | 2,975 | 0.1616 | 0.6463 | 0.4774 |
| mlp_signal__crl_best | focal__motor2 | 2,734 | 0.2196 | 0.8784 | 0.7831 |
| mlp_signal__crl_best | focal__mustang0528 | 10,643 | 0.1238 | 0.4952 | 0.3290 |
| mlp_signal__crl_best | focal__pickup2 | 2,548 | 0.0903 | 0.3613 | 0.2205 |
| mlp_signal__crl_best | focal__scooter2 | 2,762 | 0.1387 | 0.5549 | 0.3840 |
| mlp_signal__crl_best | focal__tesla2 | 2,694 | 0.0779 | 0.3116 | 0.1845 |
| mlp_signal__crl_best | focal__walk2 | 4,407 | 0.2188 | 0.8750 | 0.7778 |
| mlp_signal__crl_best | iobt | 1,575 | 0.3355 | 0.6709 | 0.5046 |
| mlp_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1441 | 0.5763 | 0.4048 |
| mlp_signal__crl_best | iobt__silverado0315pm | 490 | 0.2429 | 0.9714 | 0.9444 |
| mlp_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.0610 | 0.2439 | 0.1389 |
| mlp_signal__crl_best | m3nvc | 57,076 | 0.4026 | 0.8053 | 0.8519 |
| mlp_signal__crl_best | m3nvc__cx30 | 14,020 | 0.2380 | 0.9519 | 0.9082 |
| mlp_signal__crl_best | m3nvc__gle350 | 15,317 | 0.1996 | 0.7984 | 0.6645 |
| mlp_signal__crl_best | m3nvc__miata | 13,441 | 0.2389 | 0.9554 | 0.9145 |
| mlp_signal__crl_best | m3nvc__mustang | 14,298 | 0.2423 | 0.9692 | 0.9402 |
| mlp_signal__crl_best_aux_type | full | 91,325 | 0.6077 | 0.6077 | 0.7187 |
| mlp_signal__crl_best_aux_type | focal | 32,674 | 0.4222 | 0.4222 | 0.4470 |
| mlp_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1009 | 0.4038 | 0.2530 |
| mlp_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.1501 | 0.6005 | 0.4290 |
| mlp_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.2202 | 0.8809 | 0.7872 |
| mlp_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1367 | 0.5469 | 0.3764 |
| mlp_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0854 | 0.3415 | 0.2059 |
| mlp_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1583 | 0.6333 | 0.4634 |
| mlp_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0748 | 0.2994 | 0.1760 |
| mlp_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.2246 | 0.8984 | 0.8155 |
| mlp_signal__crl_best_aux_type | iobt | 1,575 | 0.3257 | 0.6514 | 0.4851 |
| mlp_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1355 | 0.5419 | 0.3716 |
| mlp_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2443 | 0.9773 | 0.9556 |
| mlp_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0556 | 0.2222 | 0.1250 |
| mlp_signal__crl_best_aux_type | m3nvc | 57,076 | 0.4129 | 0.8258 | 0.8704 |
| mlp_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2415 | 0.9659 | 0.9340 |
| mlp_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2008 | 0.8032 | 0.6712 |
| mlp_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2428 | 0.9713 | 0.9443 |
| mlp_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2441 | 0.9765 | 0.9542 |

## Per-class type F1 on test splits

From the type head only (`downstream_best_type.pth`).

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.436 | 0.457 | 0.814 | 0.626 |
| linear_fullz__crl_best | focal | 0.437 | 0.467 | 0.528 | 0.309 |
| linear_fullz__crl_best | focal__bicycle2 | 0.175 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.759 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.814 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.696 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.488 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.355 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.454 |
| linear_fullz__crl_best | focal__walk2 | 0.784 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.368 | 0.000 | 0.805 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.407 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.973 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.166 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.903 | 0.736 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.938 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.843 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.946 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.967 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.481 | 0.491 | 0.822 | 0.630 |
| linear_fullz__crl_best_aux_type | focal | 0.482 | 0.489 | 0.462 | 0.300 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.340 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.473 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.886 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.534 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.453 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.679 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.460 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.881 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.512 | 0.000 | 0.714 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.564 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.979 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.255 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.901 | 0.742 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.931 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.862 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.947 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.954 | 0.000 |
| linear_signal__crl_best | full | 0.445 | 0.467 | 0.818 | 0.595 |
| linear_signal__crl_best | focal | 0.446 | 0.466 | 0.490 | 0.242 |
| linear_signal__crl_best | focal__bicycle2 | 0.201 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__forester2 | 0.000 | 0.000 | 0.731 | 0.000 |
| linear_signal__crl_best | focal__motor2 | 0.000 | 0.878 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.595 | 0.000 |
| linear_signal__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.366 |
| linear_signal__crl_best | focal__scooter2 | 0.000 | 0.502 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.334 |
| linear_signal__crl_best | focal__walk2 | 0.821 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best | iobt | 0.000 | 0.493 | 0.000 | 0.838 |
| linear_signal__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.540 | 0.000 | 0.000 |
| linear_signal__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.973 |
| linear_signal__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.244 | 0.000 | 0.000 |
| linear_signal__crl_best | m3nvc | 0.000 | 0.000 | 0.906 | 0.712 |
| linear_signal__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.961 | 0.000 |
| linear_signal__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.783 |
| linear_signal__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.966 | 0.000 |
| linear_signal__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.977 | 0.000 |
| linear_signal__crl_best_aux_type | full | 0.480 | 0.490 | 0.827 | 0.630 |
| linear_signal__crl_best_aux_type | focal | 0.482 | 0.488 | 0.469 | 0.307 |
| linear_signal__crl_best_aux_type | focal__bicycle2 | 0.336 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.486 | 0.000 |
| linear_signal__crl_best_aux_type | focal__motor2 | 0.000 | 0.883 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.540 | 0.000 |
| linear_signal__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.466 |
| linear_signal__crl_best_aux_type | focal__scooter2 | 0.000 | 0.667 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.472 |
| linear_signal__crl_best_aux_type | focal__walk2 | 0.879 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.511 | 0.000 | 0.718 |
| linear_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.564 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.979 |
| linear_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.244 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.905 | 0.744 |
| linear_signal__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.940 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.852 |
| linear_signal__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.952 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.960 | 0.000 |
| mlp_signal__crl_best | full | 0.464 | 0.474 | 0.807 | 0.600 |
| mlp_signal__crl_best | focal | 0.466 | 0.470 | 0.427 | 0.242 |
| mlp_signal__crl_best | focal__bicycle2 | 0.306 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__forester2 | 0.000 | 0.000 | 0.646 | 0.000 |
| mlp_signal__crl_best | focal__motor2 | 0.000 | 0.878 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.495 | 0.000 |
| mlp_signal__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.361 |
| mlp_signal__crl_best | focal__scooter2 | 0.000 | 0.555 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.312 |
| mlp_signal__crl_best | focal__walk2 | 0.875 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best | iobt | 0.000 | 0.525 | 0.000 | 0.817 |
| mlp_signal__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.576 | 0.000 | 0.000 |
| mlp_signal__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.971 |
| mlp_signal__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.244 | 0.000 | 0.000 |
| mlp_signal__crl_best | m3nvc | 0.000 | 0.000 | 0.901 | 0.710 |
| mlp_signal__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.952 | 0.000 |
| mlp_signal__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.798 |
| mlp_signal__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.955 | 0.000 |
| mlp_signal__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.969 | 0.000 |
| mlp_signal__crl_best_aux_type | full | 0.488 | 0.490 | 0.827 | 0.626 |
| mlp_signal__crl_best_aux_type | focal | 0.490 | 0.490 | 0.463 | 0.245 |
| mlp_signal__crl_best_aux_type | focal__bicycle2 | 0.404 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.601 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__motor2 | 0.000 | 0.881 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.547 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.342 |
| mlp_signal__crl_best_aux_type | focal__scooter2 | 0.000 | 0.633 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.299 |
| mlp_signal__crl_best_aux_type | focal__walk2 | 0.898 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | iobt | 0.000 | 0.489 | 0.000 | 0.814 |
| mlp_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.542 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.977 |
| mlp_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.222 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.914 | 0.738 |
| mlp_signal__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.966 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.803 |
| mlp_signal__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.971 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.977 | 0.000 |
