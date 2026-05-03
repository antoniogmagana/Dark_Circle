# Full diagnostic report — 2026-05-03_15-26-22

## CRL pre-training

- frontend: `multiscale`, d_z=32, d_model=64, n_layers=2

- elapsed: 512.68 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.192563 (epoch 98) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6768 (epoch 35) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 0 | 0.0000 | 0.3998 | 0.5366 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 0 | 0.5628 | 0.4589 | 0.6681 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 21 | 0.8468 | 0.4393 | 0.7052 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 0 | 0.5590 | 0.4389 | 0.7117 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 0 | 0.0000 | 0.4753 | 0.6900 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 30 | 0.8674 | 0.6406 | 0.7466 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.0000 | 0.4561 | 0.4561 | 0.6674 |
| linear_fullz__crl_best | focal | 32,674 | 0.0000 | 0.2908 | 0.2908 | 0.3934 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.0000 | 0.1225 | 0.4898 | 0.3243 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.0000 | 0.2307 | 0.9227 | 0.8565 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.0000 | 0.0113 | 0.0451 | 0.0230 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.0000 | 0.1781 | 0.7122 | 0.5530 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.0000 | 0.0868 | 0.3473 | 0.2102 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.0000 | 0.0591 | 0.2364 | 0.1340 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.0000 | 0.1813 | 0.7251 | 0.5687 |
| linear_fullz__crl_best | iobt | 1,575 | 0.0000 | 0.1963 | 0.3926 | 0.2258 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.0000 | 0.2369 | 0.9474 | 0.9000 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.0000 | 0.4012 | 0.8024 | 0.8254 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.0000 | 0.2129 | 0.8515 | 0.7414 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.0000 | 0.2280 | 0.9121 | 0.8385 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.0000 | 0.2330 | 0.9319 | 0.8725 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.0000 | 0.2294 | 0.9176 | 0.8477 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8605 | 0.5920 | 0.5920 | 0.7030 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8238 | 0.4542 | 0.4542 | 0.4676 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.7734 | 0.0562 | 0.2249 | 0.1267 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.9244 | 0.2122 | 0.8487 | 0.7371 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7658 | 0.2169 | 0.8677 | 0.7664 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7251 | 0.1631 | 0.6522 | 0.4839 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9433 | 0.1354 | 0.5417 | 0.3714 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.9077 | 0.0500 | 0.2002 | 0.1112 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.9070 | 0.1330 | 0.5320 | 0.3624 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.7851 | 0.1963 | 0.7850 | 0.6461 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.6892 | 0.2701 | 0.5402 | 0.4191 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.6408 | 0.1083 | 0.4331 | 0.2764 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.6651 | 0.2453 | 0.9811 | 0.9630 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8919 | 0.0263 | 0.1053 | 0.0556 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8813 | 0.4049 | 0.8097 | 0.8365 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8867 | 0.2164 | 0.8655 | 0.7629 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8902 | 0.2294 | 0.9177 | 0.8479 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8499 | 0.2331 | 0.9326 | 0.8737 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8945 | 0.2311 | 0.9245 | 0.8596 |
| linear_ztype__crl_best | full | 91,325 | 0.0000 | 0.3968 | 0.3968 | 0.5290 |
| linear_ztype__crl_best | focal | 32,674 | 0.0000 | 0.2617 | 0.2617 | 0.3564 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.0000 | 0.2220 | 0.8880 | 0.7985 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.0000 | 0.1499 | 0.5997 | 0.4282 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.0000 | 0.0086 | 0.0346 | 0.0176 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.0000 | 0.0959 | 0.3838 | 0.2375 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.0000 | 0.0868 | 0.3473 | 0.2102 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.0000 | 0.0423 | 0.1693 | 0.0925 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.0000 | 0.2327 | 0.9307 | 0.8703 |
| linear_ztype__crl_best | iobt | 1,575 | 0.0000 | 0.1953 | 0.3907 | 0.2110 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.0000 | 0.2284 | 0.9135 | 0.8407 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.0000 | 0.3580 | 0.7161 | 0.6296 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.0000 | 0.1527 | 0.6108 | 0.4397 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.0000 | 0.2279 | 0.9116 | 0.8375 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.0000 | 0.1889 | 0.7557 | 0.6074 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.0000 | 0.1892 | 0.7566 | 0.6085 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.5556 | 0.4528 | 0.4528 | 0.6433 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.3875 | 0.3452 | 0.3452 | 0.4018 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.2497 | 0.0015 | 0.0062 | 0.0031 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.6273 | 0.2153 | 0.8613 | 0.7565 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2627 | 0.1234 | 0.4935 | 0.3276 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2168 | 0.1882 | 0.7528 | 0.6036 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.6852 | 0.1647 | 0.6589 | 0.4913 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.6167 | 0.0003 | 0.0010 | 0.0005 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.5025 | 0.1714 | 0.6855 | 0.5215 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.1355 | 0.0979 | 0.3915 | 0.2434 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.0905 | 0.1649 | 0.3299 | 0.2649 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.0180 | 0.0140 | 0.0558 | 0.0287 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.0851 | 0.2481 | 0.9925 | 0.9852 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.3729 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.6390 | 0.3810 | 0.7619 | 0.7823 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.6698 | 0.1951 | 0.7802 | 0.6396 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.6474 | 0.2374 | 0.9496 | 0.9041 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.5612 | 0.2216 | 0.8864 | 0.7960 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.6669 | 0.2183 | 0.8734 | 0.7752 |
| mlp_ztype__crl_best | full | 91,325 | 0.8440 | 0.4348 | 0.4348 | 0.7043 |
| mlp_ztype__crl_best | focal | 32,674 | 0.7862 | 0.2446 | 0.2446 | 0.4145 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.6767 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.8926 | 0.2476 | 0.9904 | 0.9810 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.7272 | 0.1454 | 0.5814 | 0.4099 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7232 | 0.2481 | 0.9924 | 0.9849 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.9243 | 0.0302 | 0.1210 | 0.0644 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.8847 | 0.0007 | 0.0028 | 0.0014 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.8583 | 0.0070 | 0.0279 | 0.0142 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.7094 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.6561 | 0.2114 | 0.4228 | 0.1970 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.6010 | 0.0082 | 0.0327 | 0.0166 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.6361 | 0.2134 | 0.8535 | 0.7444 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8780 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.8744 | 0.4179 | 0.8357 | 0.8725 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8802 | 0.2340 | 0.9359 | 0.8796 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8881 | 0.2122 | 0.8490 | 0.7376 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.8423 | 0.2445 | 0.9779 | 0.9567 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8824 | 0.2412 | 0.9647 | 0.9318 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.5519 | 0.4316 | 0.4316 | 0.7064 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.3825 | 0.2440 | 0.2440 | 0.4196 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.2443 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.6198 | 0.2493 | 0.9972 | 0.9944 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2586 | 0.1539 | 0.6156 | 0.4446 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2135 | 0.2498 | 0.9993 | 0.9986 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.6807 | 0.0188 | 0.0751 | 0.0390 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.6104 | 0.0009 | 0.0038 | 0.0019 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.4961 | 0.0016 | 0.0066 | 0.0033 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.1326 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.0888 | 0.1953 | 0.3906 | 0.1710 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.0150 | 0.0325 | 0.1299 | 0.0695 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.0851 | 0.1691 | 0.6765 | 0.5111 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.3729 | 0.0000 | 0.0000 | 0.0000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.6359 | 0.4084 | 0.8168 | 0.8736 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.6669 | 0.2464 | 0.9857 | 0.9719 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.6432 | 0.1847 | 0.7387 | 0.5857 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.5578 | 0.2493 | 0.9974 | 0.9949 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.6648 | 0.2469 | 0.9874 | 0.9752 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.410 | 0.020 | 0.768 | 0.627 |
| linear_fullz__crl_best | focal | 0.425 | 0.023 | 0.506 | 0.208 |
| linear_fullz__crl_best | focal__bicycle2 | 0.490 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.923 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.045 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.712 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.347 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.236 |
| linear_fullz__crl_best | focal__walk2 | 0.725 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.785 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.947 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.873 | 0.731 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.852 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.912 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.932 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.918 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.456 | 0.481 | 0.794 | 0.638 |
| linear_fullz__crl_best_aux_type | focal | 0.458 | 0.494 | 0.527 | 0.338 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.225 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.849 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.868 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.652 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.542 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.200 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.532 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.785 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.382 | 0.000 | 0.698 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.433 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.981 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.105 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.881 | 0.738 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.866 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.918 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.933 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.924 | 0.000 |
| linear_ztype__crl_best | full | 0.322 | 0.015 | 0.615 | 0.635 |
| linear_ztype__crl_best | focal | 0.482 | 0.018 | 0.338 | 0.209 |
| linear_ztype__crl_best | focal__bicycle2 | 0.888 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.600 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.035 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.384 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.347 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.169 |
| linear_ztype__crl_best | focal__walk2 | 0.931 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.781 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.913 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.703 | 0.729 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.611 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.912 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.756 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.757 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.227 | 0.254 | 0.750 | 0.580 |
| linear_ztype__crl_best_aux_type | focal | 0.227 | 0.288 | 0.545 | 0.321 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.006 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.861 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.493 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.753 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.659 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.001 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.685 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.392 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.046 | 0.000 | 0.614 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.056 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.993 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.832 | 0.692 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.780 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.950 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.886 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.873 | 0.000 |
| mlp_ztype__crl_best | full | 0.000 | 0.304 | 0.792 | 0.643 |
| mlp_ztype__crl_best | focal | 0.000 | 0.347 | 0.561 | 0.070 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.990 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.581 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.992 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.121 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.003 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.028 |
| mlp_ztype__crl_best | focal__walk2 | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.027 | 0.000 | 0.819 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.033 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.854 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.913 | 0.758 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.936 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.849 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.978 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.965 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.000 | 0.335 | 0.800 | 0.592 |
| mlp_ztype__crl_best_aux_type | focal | 0.000 | 0.371 | 0.564 | 0.041 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.997 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.616 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.999 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.075 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.004 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.007 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.108 | 0.000 | 0.673 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.130 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.676 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.919 | 0.715 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.986 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.739 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.997 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.987 | 0.000 |
