# Full diagnostic report — 2026-05-04_05-49-26

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 437.82 min

- total epochs recorded: 78

- **best val_ref_elbo:** 4.459985 (epoch 52) → `crl_best.pth`

- **best val_aux_type_f1:** 0.5115 (epoch 10) → `crl_best_aux_type.pth`


## Phase 2 — probes (selected by max val F1, per head)

Each probe trains both heads jointly with two independent optimizers and saves two checkpoints: the presence ckpt is the epoch with max `val_pres_f1`, the type ckpt is the epoch with max `val_type_f1`. These epochs may differ.

### Presence head

| run | probe | ckpt | best_epoch | val_pres_f1 | val_pres_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 17 | 0.7432 | 0.6589 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 23 | 0.8118 | 0.7431 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 31 | 0.7536 | 0.6623 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 28 | 0.8121 | 0.7434 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 48 | 0.7462 | 0.6600 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 18 | 0.8086 | 0.7405 |

### Type head

| run | probe | ckpt | best_epoch | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 4 | 0.4871 | 0.6142 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 9 | 0.4959 | 0.6244 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 1 | 0.4711 | 0.6153 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 7 | 0.5237 | 0.6289 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 26 | 0.4929 | 0.5713 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 2 | 0.5099 | 0.5813 |

## Phase 3 — test evals

Each eval row is from a single head's checkpoint: presence rows come from `downstream_best_pres.pth`, type rows from `downstream_best_type.pth`. Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt, m3nvc, per-vehicle) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

### Presence head — test pres_f1 by split

| run | split | n_windows | pres_f1 |
|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.7575 |
| linear_fullz__crl_best | focal | 32,674 | 0.8274 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.7697 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.9114 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.8390 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.7463 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.9118 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.8670 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.8611 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.8853 |
| linear_fullz__crl_best | iobt | 1,575 | 0.8167 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.8498 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.7500 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.8000 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.7029 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.6576 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.7581 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.6932 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.6910 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8085 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.7874 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6645 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.8821 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7767 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7512 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9019 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8651 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8026 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.7327 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.5972 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.6135 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5263 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6457 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8239 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8020 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8380 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8295 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8235 |
| linear_ztype__crl_best | full | 91,325 | 0.7537 |
| linear_ztype__crl_best | focal | 32,674 | 0.8256 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7610 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.9103 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.8328 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7472 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.9113 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.8651 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.8567 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.8851 |
| linear_ztype__crl_best | iobt | 1,575 | 0.8116 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.8432 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7510 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.7899 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.6974 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.6502 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.7550 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.6847 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.6872 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8115 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.7904 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6724 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8830 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7828 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7509 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9042 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8669 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8072 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.7407 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.6019 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.6186 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5281 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6533 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8269 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8054 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8405 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8331 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8261 |
| mlp_ztype__crl_best | full | 91,325 | 0.7669 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8318 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7866 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.9123 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.8606 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7448 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.9140 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.8702 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.8717 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.8854 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.8258 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.8595 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7438 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8458 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.7166 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.6748 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.7686 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.7104 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.7024 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8118 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.7913 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6727 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8832 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7837 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7521 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9042 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8663 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8081 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.7443 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.6069 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.6245 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5319 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6563 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8268 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8059 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8404 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8327 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8259 |

### Type head — test type macro_f1 by split

| run | split | n_windows | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.4452 | 0.4452 | 0.5191 |
| linear_fullz__crl_best | focal | 32,674 | 0.3002 | 0.3002 | 0.3558 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.1625 | 0.6501 | 0.4816 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.0023 | 0.0092 | 0.0046 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.2044 | 0.8175 | 0.6914 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.0597 | 0.2388 | 0.1356 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.0182 | 0.0729 | 0.0378 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.2102 | 0.8409 | 0.7255 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.0453 | 0.1811 | 0.0996 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.2255 | 0.9020 | 0.8215 |
| linear_fullz__crl_best | iobt | 1,575 | 0.0973 | 0.1947 | 0.1784 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0959 | 0.3834 | 0.2372 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.0377 | 0.1507 | 0.0815 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.0414 | 0.1656 | 0.0903 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.2692 | 0.5383 | 0.6152 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.2177 | 0.8707 | 0.7710 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.1179 | 0.4717 | 0.3086 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.2147 | 0.8588 | 0.7526 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.2002 | 0.8008 | 0.6677 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.4706 | 0.4706 | 0.5279 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.3136 | 0.3136 | 0.3617 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1880 | 0.7519 | 0.6025 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.0027 | 0.0108 | 0.0054 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.1989 | 0.7955 | 0.6604 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0808 | 0.3232 | 0.1927 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0435 | 0.1742 | 0.0954 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1741 | 0.6966 | 0.5345 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0433 | 0.1733 | 0.0949 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.2177 | 0.8706 | 0.7708 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.1880 | 0.3760 | 0.3192 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1230 | 0.4920 | 0.3263 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.1301 | 0.5205 | 0.3519 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0921 | 0.3683 | 0.2257 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.2932 | 0.5863 | 0.6223 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.1933 | 0.7731 | 0.6301 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1769 | 0.7076 | 0.5475 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2141 | 0.8564 | 0.7488 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.1832 | 0.7328 | 0.5782 |
| linear_ztype__crl_best | full | 91,325 | 0.4418 | 0.4418 | 0.5591 |
| linear_ztype__crl_best | focal | 32,674 | 0.3357 | 0.3357 | 0.3512 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.0916 | 0.3662 | 0.2241 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.0185 | 0.0738 | 0.0383 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.2263 | 0.9053 | 0.8269 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.0720 | 0.2880 | 0.1683 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.0955 | 0.3822 | 0.2362 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.1859 | 0.7437 | 0.5920 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.0954 | 0.3815 | 0.2357 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.2041 | 0.8162 | 0.6895 |
| linear_ztype__crl_best | iobt | 1,575 | 0.2146 | 0.4292 | 0.3611 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0844 | 0.3377 | 0.2032 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.2193 | 0.8773 | 0.7815 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.1150 | 0.4599 | 0.2986 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.2359 | 0.4717 | 0.6757 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2391 | 0.9565 | 0.9166 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.0408 | 0.1634 | 0.0890 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.2365 | 0.9460 | 0.8974 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2323 | 0.9290 | 0.8674 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.4658 | 0.4658 | 0.5676 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.3358 | 0.3358 | 0.3800 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1820 | 0.7281 | 0.5725 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.0214 | 0.0857 | 0.0448 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2109 | 0.8435 | 0.7293 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0961 | 0.3846 | 0.2381 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0452 | 0.1809 | 0.0994 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1618 | 0.6474 | 0.4786 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0452 | 0.1807 | 0.0993 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.2208 | 0.8833 | 0.7911 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.1830 | 0.3660 | 0.3304 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1279 | 0.5115 | 0.3437 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.1234 | 0.4937 | 0.3278 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.1076 | 0.4305 | 0.2743 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.2777 | 0.5554 | 0.6742 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2242 | 0.8968 | 0.8129 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1050 | 0.4201 | 0.2659 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2374 | 0.9498 | 0.9044 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2167 | 0.8670 | 0.7653 |
| mlp_ztype__crl_best | full | 91,325 | 0.4359 | 0.4359 | 0.5725 |
| mlp_ztype__crl_best | focal | 32,674 | 0.3426 | 0.3426 | 0.3964 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.1261 | 0.5044 | 0.3373 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.0228 | 0.0912 | 0.0478 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.2320 | 0.9281 | 0.8658 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.1134 | 0.4535 | 0.2933 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.0289 | 0.1156 | 0.0614 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.2129 | 0.8514 | 0.7412 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.0165 | 0.0662 | 0.0342 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.2209 | 0.8835 | 0.7913 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.2073 | 0.4145 | 0.3360 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1256 | 0.5023 | 0.3353 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.1137 | 0.4549 | 0.2944 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.1470 | 0.5882 | 0.4167 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.2329 | 0.4657 | 0.6729 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2397 | 0.9586 | 0.9206 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.0362 | 0.1450 | 0.0781 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.2360 | 0.9439 | 0.8938 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2323 | 0.9291 | 0.8676 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.4817 | 0.4817 | 0.5662 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.3085 | 0.3085 | 0.3696 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1794 | 0.7174 | 0.5593 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.0214 | 0.0857 | 0.0448 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2203 | 0.8812 | 0.7876 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0903 | 0.3613 | 0.2205 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0225 | 0.0898 | 0.0470 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1654 | 0.6616 | 0.4943 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0137 | 0.0546 | 0.0281 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.2209 | 0.8837 | 0.7916 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.2021 | 0.4042 | 0.3927 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1579 | 0.6315 | 0.4615 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.0897 | 0.3587 | 0.2185 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.1436 | 0.5743 | 0.4028 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.3044 | 0.6088 | 0.6760 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2143 | 0.8571 | 0.7500 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1488 | 0.5951 | 0.4236 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2292 | 0.9168 | 0.8463 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2090 | 0.8359 | 0.7180 |

## Per-class type F1 on test splits

From the type head only (`downstream_best_type.pth`).

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.477 | 0.365 | 0.660 | 0.279 |
| linear_fullz__crl_best | focal | 0.504 | 0.433 | 0.168 | 0.096 |
| linear_fullz__crl_best | focal__bicycle2 | 0.650 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.009 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.818 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.239 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.073 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.841 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.181 |
| linear_fullz__crl_best | focal__walk2 | 0.902 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.294 | 0.000 | 0.096 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.383 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.151 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.166 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.743 | 0.334 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.871 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.472 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.859 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.801 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.443 | 0.392 | 0.644 | 0.402 |
| linear_fullz__crl_best_aux_type | focal | 0.474 | 0.399 | 0.240 | 0.142 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.752 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.011 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.795 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.323 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.174 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.697 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.173 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.871 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.408 | 0.000 | 0.344 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.492 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.520 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.368 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.717 | 0.456 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.773 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.708 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.856 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.733 | 0.000 |
| linear_ztype__crl_best | full | 0.447 | 0.418 | 0.728 | 0.174 |
| linear_ztype__crl_best | focal | 0.464 | 0.458 | 0.228 | 0.193 |
| linear_ztype__crl_best | focal__bicycle2 | 0.366 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.074 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.905 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.288 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.382 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.744 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.382 |
| linear_ztype__crl_best | focal__walk2 | 0.816 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.347 | 0.000 | 0.511 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.338 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.877 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.460 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.802 | 0.141 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.957 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.163 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.946 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.929 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.446 | 0.414 | 0.720 | 0.283 |
| linear_ztype__crl_best_aux_type | focal | 0.473 | 0.425 | 0.299 | 0.146 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.728 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.086 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.844 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.385 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.181 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.647 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.181 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.883 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.433 | 0.000 | 0.299 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.511 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.494 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.430 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.788 | 0.323 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.897 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.420 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.950 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.867 | 0.000 |
| mlp_ztype__crl_best | full | 0.486 | 0.401 | 0.737 | 0.120 |
| mlp_ztype__crl_best | focal | 0.508 | 0.434 | 0.352 | 0.076 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.504 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.091 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.928 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.454 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.116 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.851 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.066 |
| mlp_ztype__crl_best | focal__walk2 | 0.883 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.483 | 0.000 | 0.346 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.502 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.455 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.588 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.802 | 0.130 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.959 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.145 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.944 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.929 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.436 | 0.412 | 0.706 | 0.373 |
| mlp_ztype__crl_best_aux_type | focal | 0.462 | 0.423 | 0.283 | 0.066 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.717 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.086 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.881 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.361 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.090 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.662 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.055 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.884 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.541 | 0.000 | 0.267 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.631 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.359 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.574 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.778 | 0.440 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.857 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.595 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.917 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.836 | 0.000 |
