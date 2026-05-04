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
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.4667 | 0.0324 | 0.1297 | 0.0694 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.7836 | 0.2068 | 0.8271 | 0.7052 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.2628 | 0.2006 | 0.8022 | 0.6697 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.3591 | 0.1927 | 0.7707 | 0.6269 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.7292 | 0.1060 | 0.4239 | 0.2689 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.7670 | 0.0758 | 0.3032 | 0.1787 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.6218 | 0.0930 | 0.3719 | 0.2284 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.1650 | 0.1761 | 0.7043 | 0.5436 |
| linear_fullz__crl_best | iobt | 1,575 | 0.4238 | 0.2888 | 0.5776 | 0.3727 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.3589 | 0.0878 | 0.3512 | 0.2130 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.4581 | 0.2404 | 0.9615 | 0.9259 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.6180 | 0.0325 | 0.1299 | 0.0694 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.8453 | 0.4093 | 0.8186 | 0.8641 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.8441 | 0.2396 | 0.9582 | 0.9197 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.8601 | 0.2007 | 0.8029 | 0.6707 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.8259 | 0.2406 | 0.9623 | 0.9273 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.8482 | 0.2447 | 0.9790 | 0.9589 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.6848 | 0.6266 | 0.6266 | 0.7272 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.1290 | 0.4351 | 0.4351 | 0.4796 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0482 | 0.0864 | 0.3455 | 0.2088 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.2923 | 0.1475 | 0.5901 | 0.4185 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.0285 | 0.2214 | 0.8857 | 0.7948 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0819 | 0.1477 | 0.5910 | 0.4195 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.3065 | 0.0865 | 0.3460 | 0.2092 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.2342 | 0.1664 | 0.6658 | 0.4990 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1627 | 0.0770 | 0.3082 | 0.1822 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.0074 | 0.2198 | 0.8790 | 0.7841 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.3326 | 0.3239 | 0.6478 | 0.5000 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.0852 | 0.1406 | 0.5624 | 0.3912 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7244 | 0.2448 | 0.9792 | 0.9593 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.3200 | 0.0610 | 0.2439 | 0.1389 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8449 | 0.4095 | 0.8191 | 0.8538 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8384 | 0.2308 | 0.9233 | 0.8575 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8769 | 0.2170 | 0.8679 | 0.7666 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8166 | 0.2356 | 0.9425 | 0.8913 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8397 | 0.2380 | 0.9520 | 0.9083 |
| linear_signal__crl_best | full | 88,631 | 0.8245 | 0.5789 | 0.5789 | 0.7316 |
| linear_signal__crl_best | focal | 29,980 | 0.6733 | 0.4215 | 0.4215 | 0.4796 |
| linear_signal__crl_best | focal__bicycle2 | 3,911 | 0.7388 | 0.0324 | 0.1297 | 0.0694 |
| linear_signal__crl_best | focal__forester2 | 2,975 | 0.9026 | 0.2122 | 0.8489 | 0.7375 |
| linear_signal__crl_best | focal__motor2 | 2,734 | 0.5059 | 0.1942 | 0.7767 | 0.6349 |
| linear_signal__crl_best | focal__mustang0528 | 10,643 | 0.5299 | 0.1962 | 0.7848 | 0.6458 |
| linear_signal__crl_best | focal__pickup2 | 2,548 | 0.8725 | 0.1035 | 0.4139 | 0.2609 |
| linear_signal__crl_best | focal__scooter2 | 2,762 | 0.8699 | 0.0631 | 0.2525 | 0.1445 |
| linear_signal__crl_best | focal__tesla2 | 2,694 | 0.8028 | 0.0879 | 0.3516 | 0.2133 |
| linear_signal__crl_best | focal__walk2 | 4,407 | 0.3837 | 0.1746 | 0.6985 | 0.5367 |
| linear_signal__crl_best | iobt | 1,575 | 0.7085 | 0.2826 | 0.5652 | 0.3532 |
| linear_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.6672 | 0.0784 | 0.3134 | 0.1858 |
| linear_signal__crl_best | iobt__silverado0315pm | 490 | 0.7757 | 0.2399 | 0.9595 | 0.9222 |
| linear_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.7666 | 0.0263 | 0.1053 | 0.0556 |
| linear_signal__crl_best | m3nvc | 57,076 | 0.8819 | 0.4082 | 0.8164 | 0.8640 |
| linear_signal__crl_best | m3nvc__cx30 | 14,020 | 0.8784 | 0.2406 | 0.9624 | 0.9276 |
| linear_signal__crl_best | m3nvc__gle350 | 15,317 | 0.8811 | 0.1978 | 0.7911 | 0.6545 |
| linear_signal__crl_best | m3nvc__miata | 13,441 | 0.8714 | 0.2414 | 0.9657 | 0.9338 |
| linear_signal__crl_best | m3nvc__mustang | 14,298 | 0.8961 | 0.2452 | 0.9809 | 0.9624 |
| linear_signal__crl_best_aux_type | full | 88,631 | 0.7472 | 0.6303 | 0.6303 | 0.7330 |
| linear_signal__crl_best_aux_type | focal | 29,980 | 0.2658 | 0.4293 | 0.4293 | 0.4788 |
| linear_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1133 | 0.0899 | 0.3597 | 0.2193 |
| linear_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.5419 | 0.1484 | 0.5937 | 0.4222 |
| linear_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.0747 | 0.2231 | 0.8924 | 0.8057 |
| linear_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2119 | 0.1457 | 0.5829 | 0.4114 |
| linear_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.4809 | 0.0741 | 0.2964 | 0.1740 |
| linear_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.4538 | 0.1690 | 0.6759 | 0.5105 |
| linear_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.2895 | 0.0611 | 0.2444 | 0.1392 |
| linear_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.0287 | 0.2208 | 0.8833 | 0.7909 |
| linear_signal__crl_best_aux_type | iobt | 1,575 | 0.4273 | 0.3363 | 0.6726 | 0.5093 |
| linear_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.2158 | 0.1441 | 0.5763 | 0.4048 |
| linear_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7521 | 0.2448 | 0.9792 | 0.9593 |
| linear_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.4811 | 0.0636 | 0.2545 | 0.1458 |
| linear_signal__crl_best_aux_type | m3nvc | 57,076 | 0.8818 | 0.4130 | 0.8259 | 0.8628 |
| linear_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8813 | 0.2350 | 0.9399 | 0.8867 |
| linear_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8805 | 0.2131 | 0.8524 | 0.7427 |
| linear_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8674 | 0.2381 | 0.9522 | 0.9088 |
| linear_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8975 | 0.2403 | 0.9612 | 0.9253 |
| mlp_signal__crl_best | full | 88,631 | 0.7842 | 0.6021 | 0.6021 | 0.7291 |
| mlp_signal__crl_best | focal | 29,980 | 0.5665 | 0.4306 | 0.4306 | 0.4727 |
| mlp_signal__crl_best | focal__bicycle2 | 3,911 | 0.5896 | 0.0481 | 0.1926 | 0.1065 |
| mlp_signal__crl_best | focal__forester2 | 2,975 | 0.8580 | 0.1945 | 0.7780 | 0.6367 |
| mlp_signal__crl_best | focal__motor2 | 2,734 | 0.3875 | 0.2094 | 0.8377 | 0.7207 |
| mlp_signal__crl_best | focal__mustang0528 | 10,643 | 0.3787 | 0.1679 | 0.6715 | 0.5055 |
| mlp_signal__crl_best | focal__pickup2 | 2,548 | 0.8180 | 0.1058 | 0.4233 | 0.2685 |
| mlp_signal__crl_best | focal__scooter2 | 2,762 | 0.8266 | 0.0955 | 0.3822 | 0.2362 |
| mlp_signal__crl_best | focal__tesla2 | 2,694 | 0.7175 | 0.0948 | 0.3793 | 0.2341 |
| mlp_signal__crl_best | focal__walk2 | 4,407 | 0.2021 | 0.2019 | 0.8076 | 0.6772 |
| mlp_signal__crl_best | iobt | 1,575 | 0.5524 | 0.3097 | 0.6195 | 0.4238 |
| mlp_signal__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.4920 | 0.1110 | 0.4442 | 0.2855 |
| mlp_signal__crl_best | iobt__silverado0315pm | 490 | 0.6188 | 0.2419 | 0.9675 | 0.9370 |
| mlp_signal__crl_best | iobt__warhog_nolineofsight | 190 | 0.6853 | 0.0443 | 0.1772 | 0.0972 |
| mlp_signal__crl_best | m3nvc | 57,076 | 0.8668 | 0.4096 | 0.8192 | 0.8619 |
| mlp_signal__crl_best | m3nvc__cx30 | 14,020 | 0.8668 | 0.2378 | 0.9513 | 0.9072 |
| mlp_signal__crl_best | m3nvc__gle350 | 15,317 | 0.8831 | 0.2047 | 0.8189 | 0.6934 |
| mlp_signal__crl_best | m3nvc__miata | 13,441 | 0.8452 | 0.2389 | 0.9557 | 0.9152 |
| mlp_signal__crl_best | m3nvc__mustang | 14,298 | 0.8686 | 0.2434 | 0.9738 | 0.9489 |
| mlp_signal__crl_best_aux_type | full | 88,631 | 0.7391 | 0.6284 | 0.6284 | 0.7288 |
| mlp_signal__crl_best_aux_type | focal | 29,980 | 0.2160 | 0.4374 | 0.4374 | 0.4773 |
| mlp_signal__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0856 | 0.0981 | 0.3924 | 0.2441 |
| mlp_signal__crl_best_aux_type | focal__forester2 | 2,975 | 0.4420 | 0.1473 | 0.5893 | 0.4177 |
| mlp_signal__crl_best_aux_type | focal__motor2 | 2,734 | 0.0518 | 0.2200 | 0.8801 | 0.7858 |
| mlp_signal__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1776 | 0.1407 | 0.5629 | 0.3917 |
| mlp_signal__crl_best_aux_type | focal__pickup2 | 2,548 | 0.4041 | 0.0933 | 0.3732 | 0.2294 |
| mlp_signal__crl_best_aux_type | focal__scooter2 | 2,762 | 0.3826 | 0.1577 | 0.6307 | 0.4606 |
| mlp_signal__crl_best_aux_type | focal__tesla2 | 2,694 | 0.2412 | 0.0762 | 0.3048 | 0.1798 |
| mlp_signal__crl_best_aux_type | focal__walk2 | 4,407 | 0.0232 | 0.2238 | 0.8954 | 0.8106 |
| mlp_signal__crl_best_aux_type | iobt | 1,575 | 0.4206 | 0.3197 | 0.6394 | 0.4777 |
| mlp_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.2036 | 0.1326 | 0.5305 | 0.3610 |
| mlp_signal__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7515 | 0.2438 | 0.9753 | 0.9519 |
| mlp_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.4660 | 0.0556 | 0.2222 | 0.1250 |
| mlp_signal__crl_best_aux_type | m3nvc | 57,076 | 0.8812 | 0.4109 | 0.8217 | 0.8579 |
| mlp_signal__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8817 | 0.2328 | 0.9312 | 0.8712 |
| mlp_signal__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8808 | 0.2146 | 0.8584 | 0.7520 |
| mlp_signal__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8657 | 0.2368 | 0.9472 | 0.8996 |
| mlp_signal__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8960 | 0.2395 | 0.9580 | 0.9193 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.406 | 0.465 | 0.827 | 0.639 |
| linear_fullz__crl_best | focal | 0.407 | 0.484 | 0.579 | 0.229 |
| linear_fullz__crl_best | focal__bicycle2 | 0.130 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.827 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.802 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.771 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.424 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.303 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.372 |
| linear_fullz__crl_best | focal__walk2 | 0.704 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.315 | 0.000 | 0.840 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.351 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.962 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.130 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.909 | 0.728 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.958 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.803 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.962 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.979 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.502 | 0.519 | 0.821 | 0.664 |
| linear_fullz__crl_best_aux_type | focal | 0.504 | 0.520 | 0.506 | 0.210 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.345 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.590 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.886 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.591 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.346 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.666 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.308 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.879 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.510 | 0.000 | 0.786 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.562 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.979 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.244 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.898 | 0.740 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.923 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.868 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.943 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.952 | 0.000 |
| linear_signal__crl_best | full | 0.404 | 0.448 | 0.826 | 0.639 |
| linear_signal__crl_best | focal | 0.404 | 0.470 | 0.578 | 0.234 |
| linear_signal__crl_best | focal__bicycle2 | 0.130 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__forester2 | 0.000 | 0.000 | 0.849 | 0.000 |
| linear_signal__crl_best | focal__motor2 | 0.000 | 0.777 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.785 | 0.000 |
| linear_signal__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.414 |
| linear_signal__crl_best | focal__scooter2 | 0.000 | 0.253 | 0.000 | 0.000 |
| linear_signal__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.352 |
| linear_signal__crl_best | focal__walk2 | 0.699 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best | iobt | 0.000 | 0.279 | 0.000 | 0.851 |
| linear_signal__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.313 | 0.000 | 0.000 |
| linear_signal__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.960 |
| linear_signal__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.105 | 0.000 | 0.000 |
| linear_signal__crl_best | m3nvc | 0.000 | 0.000 | 0.910 | 0.723 |
| linear_signal__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.962 | 0.000 |
| linear_signal__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.791 |
| linear_signal__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.966 | 0.000 |
| linear_signal__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.981 | 0.000 |
| linear_signal__crl_best_aux_type | full | 0.503 | 0.519 | 0.826 | 0.672 |
| linear_signal__crl_best_aux_type | focal | 0.505 | 0.519 | 0.499 | 0.194 |
| linear_signal__crl_best_aux_type | focal__bicycle2 | 0.360 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.594 | 0.000 |
| linear_signal__crl_best_aux_type | focal__motor2 | 0.000 | 0.892 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.583 | 0.000 |
| linear_signal__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.296 |
| linear_signal__crl_best_aux_type | focal__scooter2 | 0.000 | 0.676 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.244 |
| linear_signal__crl_best_aux_type | focal__walk2 | 0.883 | 0.000 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.523 | 0.000 | 0.822 |
| linear_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.576 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.979 |
| linear_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.255 | 0.000 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.906 | 0.746 |
| linear_signal__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.940 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.852 |
| linear_signal__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.952 | 0.000 |
| linear_signal__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.961 | 0.000 |
| mlp_signal__crl_best | full | 0.456 | 0.485 | 0.824 | 0.643 |
| mlp_signal__crl_best | focal | 0.457 | 0.496 | 0.542 | 0.227 |
| mlp_signal__crl_best | focal__bicycle2 | 0.193 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__forester2 | 0.000 | 0.000 | 0.778 | 0.000 |
| mlp_signal__crl_best | focal__motor2 | 0.000 | 0.838 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.671 | 0.000 |
| mlp_signal__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.423 |
| mlp_signal__crl_best | focal__scooter2 | 0.000 | 0.382 | 0.000 | 0.000 |
| mlp_signal__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.379 |
| mlp_signal__crl_best | focal__walk2 | 0.808 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best | iobt | 0.000 | 0.401 | 0.000 | 0.838 |
| mlp_signal__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.444 | 0.000 | 0.000 |
| mlp_signal__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.968 |
| mlp_signal__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.177 | 0.000 | 0.000 |
| mlp_signal__crl_best | m3nvc | 0.000 | 0.000 | 0.907 | 0.731 |
| mlp_signal__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.951 | 0.000 |
| mlp_signal__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.819 |
| mlp_signal__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.956 | 0.000 |
| mlp_signal__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.974 | 0.000 |
| mlp_signal__crl_best_aux_type | full | 0.509 | 0.515 | 0.821 | 0.669 |
| mlp_signal__crl_best_aux_type | focal | 0.511 | 0.519 | 0.486 | 0.233 |
| mlp_signal__crl_best_aux_type | focal__bicycle2 | 0.392 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.589 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__motor2 | 0.000 | 0.880 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.563 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.373 |
| mlp_signal__crl_best_aux_type | focal__scooter2 | 0.000 | 0.631 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.305 |
| mlp_signal__crl_best_aux_type | focal__walk2 | 0.895 | 0.000 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | iobt | 0.000 | 0.479 | 0.000 | 0.799 |
| mlp_signal__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.530 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.975 |
| mlp_signal__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.222 | 0.000 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.902 | 0.741 |
| mlp_signal__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.931 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.858 |
| mlp_signal__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.947 | 0.000 |
| mlp_signal__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.958 | 0.000 |
