# Ablation Pairs

Runs considered: 13

A pair matches an axis when those two runs differ in that axis AND all other tracked axes are identical. Δ = variant − baseline.

## frontend_type

| Baseline | Variant | Axis: base → variant | Δpres_f1 | Δtype_f1 | Δref_elbo |
|---|---|---|---|---|---|
| 2026-04-24_18-57 | phase_v1_diag | morlet_fused → morlet_per_sensor | -0.0308 | +0.1488 | -0.0345 |

## morlet_use_phase

_No matching pairs found._

## prior_type

_No matching pairs found._

## training_mode

| Baseline | Variant | Axis: base → variant | Δpres_f1 | Δtype_f1 | Δref_elbo |
|---|---|---|---|---|---|
| 2026-04-25_10-06 | phase_v1_diag | disentangled → vae | -0.0823 | +0.1276 | +0.3005 |
| 2026-04-24_16-17 | 2026-04-25_09-41 | contrastive → disentangled | +0.1340 | +0.0749 | — |
| 2026-04-24_16-17 | v3_lowfreq | contrastive → disentangled | +0.1093 | +0.2470 | — |
| 2026-04-24_16-17 | 2026-04-24_16-31 | contrastive → vae | +0.1318 | +0.0689 | — |
| 2026-04-24_16-17 | 2026-04-28_23-12-59 | contrastive → vae | +0.1258 | +0.2322 | — |
| 2026-04-24_16-17 | 2026-04-29_13-26-17 | contrastive → vae | +0.1223 | +0.2560 | — |
| 2026-04-24_16-17 | filesplit_v2 | contrastive → vae | +0.1405 | +0.1219 | — |
| 2026-04-24_16-17 | v2 | contrastive → vae | +0.1248 | +0.2053 | — |
| 2026-04-24_16-17 | v3_lowfreq | contrastive → vae | +0.1231 | +0.2536 | — |
| 2026-04-25_09-41 | 2026-04-24_16-31 | disentangled → vae | -0.0022 | -0.0060 | +0.0299 |
| 2026-04-25_09-41 | 2026-04-28_23-12-59 | disentangled → vae | -0.0082 | +0.1573 | +0.0703 |
| 2026-04-25_09-41 | 2026-04-29_13-26-17 | disentangled → vae | -0.0117 | +0.1811 | +0.0488 |
| 2026-04-25_09-41 | filesplit_v2 | disentangled → vae | +0.0065 | +0.0470 | +0.0193 |
| 2026-04-25_09-41 | v2 | disentangled → vae | -0.0092 | +0.1304 | +0.0362 |
| 2026-04-25_09-41 | v3_lowfreq | disentangled → vae | -0.0109 | +0.1787 | +0.0591 |
| v3_lowfreq | 2026-04-24_16-31 | disentangled → vae | +0.0225 | -0.1781 | +0.0664 |
| v3_lowfreq | 2026-04-28_23-12-59 | disentangled → vae | +0.0165 | -0.0148 | +0.1068 |
| v3_lowfreq | 2026-04-29_13-26-17 | disentangled → vae | +0.0130 | +0.0090 | +0.0854 |
| v3_lowfreq | filesplit_v2 | disentangled → vae | +0.0312 | -0.1251 | +0.0559 |
| v3_lowfreq | v2 | disentangled → vae | +0.0155 | -0.0417 | +0.0727 |
| v3_lowfreq | v3_lowfreq | disentangled → vae | +0.0138 | +0.0066 | +0.0956 |

## stage2

_No matching pairs found._

## morlet_learnable_w0

_No matching pairs found._
