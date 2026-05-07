# Ablation Pairs

Runs considered: 15

A pair matches an axis when those two runs differ in that axis AND all other tracked axes are identical. Δ = variant − baseline.

## frontend_type

| Baseline | Variant | Axis: base → variant | Δpres_f1 | Δtype_f1 | Δref_elbo |
|---|---|---|---|---|---|
| 2026-04-24_18-57 | phase_v1_diag | morlet_fused → morlet_per_sensor | -0.0308 | +0.1488 | -0.0345 |
| 2026-05-04_05-49-26 | 2026-04-24_16-31 | morlet_per_sensor → multiscale | +0.1405 | -0.0153 | -4.2863 |
| 2026-05-04_05-49-26 | 2026-04-28_23-12-59 | morlet_per_sensor → multiscale | +0.1345 | +0.1480 | -4.2460 |
| 2026-05-04_05-49-26 | 2026-04-29_13-26-17 | morlet_per_sensor → multiscale | +0.1310 | +0.1718 | -4.2674 |
| 2026-05-04_05-49-26 | 2026-05-03_05-02-44 | morlet_per_sensor → multiscale | +0.1193 | +0.1520 | -3.9074 |
| 2026-05-04_05-49-26 | filesplit_v2 | morlet_per_sensor → multiscale | +0.1492 | +0.0377 | -4.2969 |
| 2026-05-04_05-49-26 | v2 | morlet_per_sensor → multiscale | +0.1335 | +0.1211 | -4.2800 |
| 2026-05-04_05-49-26 | v3_lowfreq | morlet_per_sensor → multiscale | +0.1318 | +0.1694 | -4.2572 |

## morlet_use_phase

| Baseline | Variant | Axis: base → variant | Δpres_f1 | Δtype_f1 | Δref_elbo |
|---|---|---|---|---|---|
| 2026-05-04_05-49-26 | phase_v1_diag | False → True | +0.0452 | +0.0106 | -2.1831 |

## prior_type

_No matching pairs found._

## training_mode

| Baseline | Variant | Axis: base → variant | Δpres_f1 | Δtype_f1 | Δref_elbo |
|---|---|---|---|---|---|
| 2026-04-25_10-06 | phase_v1_diag | disentangled → vae | -0.0823 | +0.1276 | +0.3005 |
| 2026-04-24_16-17 | 2026-04-25_09-41 | contrastive → disentangled | +0.1340 | +0.0749 | — |
| 2026-04-24_16-17 | 2026-05-03_05-03-14 | contrastive → disentangled | +0.0914 | +0.2433 | — |
| 2026-04-24_16-17 | v3_lowfreq | contrastive → disentangled | +0.1093 | +0.2470 | — |
| 2026-04-24_16-17 | 2026-04-24_16-31 | contrastive → vae | +0.1318 | +0.0689 | — |
| 2026-04-24_16-17 | 2026-04-28_23-12-59 | contrastive → vae | +0.1258 | +0.2322 | — |
| 2026-04-24_16-17 | 2026-04-29_13-26-17 | contrastive → vae | +0.1223 | +0.2560 | — |
| 2026-04-24_16-17 | 2026-05-03_05-02-44 | contrastive → vae | +0.1106 | +0.2362 | — |
| 2026-04-24_16-17 | filesplit_v2 | contrastive → vae | +0.1405 | +0.1219 | — |
| 2026-04-24_16-17 | v2 | contrastive → vae | +0.1248 | +0.2053 | — |
| 2026-04-24_16-17 | v3_lowfreq | contrastive → vae | +0.1231 | +0.2536 | — |
| 2026-04-25_09-41 | 2026-04-24_16-31 | disentangled → vae | -0.0022 | -0.0060 | +0.0299 |
| 2026-04-25_09-41 | 2026-04-28_23-12-59 | disentangled → vae | -0.0082 | +0.1573 | +0.0703 |
| 2026-04-25_09-41 | 2026-04-29_13-26-17 | disentangled → vae | -0.0117 | +0.1811 | +0.0488 |
| 2026-04-25_09-41 | 2026-05-03_05-02-44 | disentangled → vae | -0.0234 | +0.1613 | +0.4089 |
| 2026-04-25_09-41 | filesplit_v2 | disentangled → vae | +0.0065 | +0.0470 | +0.0193 |
| 2026-04-25_09-41 | v2 | disentangled → vae | -0.0092 | +0.1304 | +0.0362 |
| 2026-04-25_09-41 | v3_lowfreq | disentangled → vae | -0.0109 | +0.1787 | +0.0591 |
| 2026-05-03_05-03-14 | 2026-04-24_16-31 | disentangled → vae | +0.0404 | -0.1744 | -0.0787 |
| 2026-05-03_05-03-14 | 2026-04-28_23-12-59 | disentangled → vae | +0.0344 | -0.0111 | -0.0384 |
| 2026-05-03_05-03-14 | 2026-04-29_13-26-17 | disentangled → vae | +0.0309 | +0.0127 | -0.0598 |
| 2026-05-03_05-03-14 | 2026-05-03_05-02-44 | disentangled → vae | +0.0192 | -0.0071 | +0.3002 |
| 2026-05-03_05-03-14 | filesplit_v2 | disentangled → vae | +0.0491 | -0.1214 | -0.0893 |
| 2026-05-03_05-03-14 | v2 | disentangled → vae | +0.0334 | -0.0380 | -0.0724 |
| 2026-05-03_05-03-14 | v3_lowfreq | disentangled → vae | +0.0317 | +0.0103 | -0.0496 |
| v3_lowfreq | 2026-04-24_16-31 | disentangled → vae | +0.0225 | -0.1781 | +0.0664 |
| v3_lowfreq | 2026-04-28_23-12-59 | disentangled → vae | +0.0165 | -0.0148 | +0.1068 |
| v3_lowfreq | 2026-04-29_13-26-17 | disentangled → vae | +0.0130 | +0.0090 | +0.0854 |
| v3_lowfreq | 2026-05-03_05-02-44 | disentangled → vae | +0.0013 | -0.0108 | +0.4454 |
| v3_lowfreq | filesplit_v2 | disentangled → vae | +0.0312 | -0.1251 | +0.0559 |
| v3_lowfreq | v2 | disentangled → vae | +0.0155 | -0.0417 | +0.0727 |
| v3_lowfreq | v3_lowfreq | disentangled → vae | +0.0138 | +0.0066 | +0.0956 |

## stage2

_No matching pairs found._

## morlet_learnable_w0

_No matching pairs found._
