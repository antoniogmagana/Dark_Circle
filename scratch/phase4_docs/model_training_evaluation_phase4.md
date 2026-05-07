# Techniques for Tactical Edge Data Support to the Army Research Laboratory's Live, Virtual, and Constructive (LVC) Toolkit

## Model Training Evaluation (Phase 4)

**Artificial Intelligence (AI) Technician Capstone Group 5**

Students: Brandon Taylor, Antonio Magana, Larry Parrotte, John Tomaselli — Carnegie Mellon University

Mentors: Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Mr. Henry Reimert — DEVCOM Army Research Laboratory

*This is the first formal Model Training Evaluation deliverable for the project; Phase 3 had no separate document of this kind. Numbers throughout this document come from the cited source files (each saved run's `meta.json`, `report.json`, `crl_metrics.csv`, and `downstream_metrics.csv`, plus the bundle catalog files at `inference-engine/{detect,classify}-bundles/`). No number is paraphrased from memory or summarized from intermediate analysis files.*

---

## Purpose

This document assesses, compares, and communicates the performance of the trained models that ship in the Phase 4 inference engine. The Phase 4 system ships **two** independently-trained CRL models, one per inference pod, so the document covers both: a multiscale-VAE backbone deployed in the detect pod, and a multiscale-disentangled backbone deployed in the classify pod. Both models contribute to a single end-to-end pipeline (window → presence → type) that is the actual product, so the evaluation looks at each model individually and at the trade that motivated shipping two of them.

## Training Objective

The two shipping models share the same encoder topology but have different training objectives, which is the central modeling decision of this phase.

**Detect side: variational ELBO with auxiliary supervision.** The detect-side training objective is a β-annealed evidence lower bound on the reconstruction of the fused frontend features, with three auxiliary losses attached: a presence classifier, a vehicle-type classifier, and an intervention classifier on the env block of the latent. The full loss is `L = L_recon + β·L_KL + λ_interv·L_interv + λ_aux_pres·L_aux_pres + λ_aux_type·L_aux_type` with all λ weights set to 1.0 except `λ_aux_prox=0.1` (`crl_vehicle/training_modes/vae_mode.py:34-209`, weights from the run's `report.json` `crl.config`). Why this objective: a representation-learning backbone supports a downstream presence head while the β-invariant `val_ref_elbo` remains a comparable checkpoint metric across the non-stationary β-annealing schedule. The auxiliary supervision biases the latent toward a partition (`CausalLatentSpace` with named D_PRES, D_TYPE, D_PROX, D_ENV, D_FREE blocks) that downstream linear probes can exploit cleanly.

**Classify side: disentangled ELBO over a 2-block signal/env latent.** The classify-side training objective adds three disentanglement losses to the ELBO: a cross-modal alignment loss between the audio and seismic signal blocks (`λ_align=1.0`), an env temporal-stability loss penalizing fast change in the env block across nearby windows (`λ_stab=0.1`), and a signal intervention-invariance loss penalizing change in the signal block under synthetic noise interventions (`λ_interv_inv=1.0`) (`crl_vehicle/training_modes/disentangled_mode.py:45-82`, weights from `crl/meta.json`). The latent partition is a `SplitLatentSpace` with `d_z=24` and `d_signal=12`, so the first 12 dims are the signal block and the remaining 12 are env. Why this objective: forcing vehicle-type-relevant information into a small known sub-block (z[0:12]) lets a single-layer linear probe achieve competitive type F1, which both keeps the deployed type head trivial to ship and serves as evidence that the disentanglement actually localizes the type signal as intended.

The same auxiliary heads (presence, type) are attached on both runs, but the disentangled side's heads are owned by the training mode (sized to `d_signal=12`) rather than by the model (which would size them to `D_PRES=4` and `D_TYPE=6` from the causal partition).

## Datasets

Both shipping runs draw from the same combined dataset described in `Dark_Circle/DATASET_CONSTRUCTION.md`. The dataset combines three sources:

- **M3NVC**: 4 vehicles (Mazda CX-30, Mercedes-Benz GLE 350, Ford Mustang, Mazda MX-5), 6 unique deployment scenes, 18.26 hours of recording. Acoustic at 1600 Hz, seismic at 200 Hz.
- **MOD_vehicle (IOBT)**: 15 vehicle/object instances, 42.5 hours of recording. Acoustic at 16 kHz, seismic at 100 Hz, 3-axis accelerometer at 100 Hz.
- **MOD_vehicle (FOCAL)**: 12 vehicle/object instances, 31.8 hours of recording. Same modality and rates as IOBT.

Raw stats: 333,216 raw 1-second windows across the three sources combined; 125,000 windows retained after cleaning and deduplication.

**Splits**. Both shipping runs use the ID-split schema defined in `crl-train/docs/superpowers/specs/2026-04-25-id-split-schema-design.md` (`use_id_split=true` in both meta.json files). The split partitions windows by vehicle/scene identity rather than by file placement, so the same vehicle does not appear in both train and validation splits. Globally the split is 70/15/15 train/val/test with class balance verified across all three partitions before training begins.

**Window construction**. Audio is resampled to 16 kHz; seismic is left at 100 Hz. The window length is one second; per-window DC removal (mean subtraction) is the only preprocessing applied before the data enters the frontend. Per-modality bit-depth handling (audio 16-bit, seismic and accelerometer 24-bit) is configured at the inference-engine ConfigMap layer and is not touched by the training pipeline.

**Augmentation**. Seven intervention noise types (white, brown, pink, green, sinusoidal, chirp, bird-chirp) are mixed in at 20% RMS of the signal during training only. The intervention augmentation is what the signal intervention-invariance loss on the classify side trains against. Evaluation windows are not augmented.

## Training Hyperparameters

The values below are quoted from each shipping run's `meta.json` (or, for the detect run which does not have a `crl/meta.json` because its directory layout pre-dates the current trainer's output convention, from `report.json` `crl.config`). The team's hyperparameter-tuning approach is ablation-driven: each candidate change runs as a single one-variable-at-a-time ablation and is compared via `compare_runs.py` and `compare_ablations.py` against a reference run, rather than via grid, random, or Bayesian search.

**Common to both shipping runs.**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate (initial) | 3 × 10⁻⁴ |
| Learning rate floor (cosine) | 1 × 10⁻⁴ |
| Cosine schedule T_max | 100 epochs |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 128 |
| DataLoader workers | 8 |
| Maximum epochs | 100 |
| Early-stop patience | 25 epochs on val_ref_elbo |
| Random seed | 42 |
| Frontend | multiscale, audio kernels [9, 19, 39, 159], seismic kernels [9, 19, 39] |
| Frontend pool stride | 16 |
| Fused token length | 32 per sensor → 64 fused |
| Encoder | 2-layer transformer, d_model=64, 4 heads, FFN width 128 |
| KL annealing | β_step=0.02, kl_floor=0.01, kl_target=0.5, recon_min_delta=1×10⁻⁴ |
| Aux weights | λ_aux_pres=1.0, λ_aux_type=1.0, λ_aux_prox=0.1, λ_interv=1.0 |

**Detect-side (VAE) specifics.**

| Parameter | Value |
|---|---|
| Latent dim d_z | 32 |
| Latent partition | CausalLatentSpace: D_PRES=4, D_TYPE=6, D_PROX=3, D_ENV=6, D_FREE=5 |
| Aux type loss | focal cross-entropy with γ=2.0 (`use_focal_type=true`) |

**Classify-side (disentangled) specifics.**

| Parameter | Value |
|---|---|
| Latent dim d_z | 24 |
| Latent partition | SplitLatentSpace: d_signal=12, d_env=12 |
| Aux type loss | standard cross-entropy (`use_focal_type=false`) |
| Cross-modal alignment weight | λ_align=1.0 |
| Env temporal stability weight | λ_stab=0.1 |
| Signal intervention-invariance weight | λ_interv_inv=1.0 |

Both runs ran for the full 100-epoch budget; neither triggered early-stop. The detect-side run's wall-clock CRL time was 512.7 minutes; the classify side took 549.3 minutes. Each run subsequently trained six small downstream probes (50 epochs each, AdamW at the same learning rate, separate optimizers for the presence and type heads, frozen CRL backbone) — these add a few minutes per probe and are not counted in the CRL elapsed time.

## Training Curves

Per-epoch training-curve data is available for the classify-side disentangled run (`crl/crl_metrics.csv`, 100 epochs × 26 columns). The detect-side VAE run was logged before the current per-epoch CSV convention was added to the trainer's saved-run layout, so for that run only the trajectory summary in `report.json` is available — the per-epoch curves cannot be reconstructed without re-running the experiment.

**Classify-side CRL pretraining.** The reference ELBO drops sharply in the first epoch (38.4 → 0.45 by epoch 5) and then refines slowly, reaching the run's best at epoch 99 (`val_ref_elbo=0.2524`). The β-anneal schedule starts at 0.02 and grows toward a hold near 0.44, but the climb is gradual: at the epoch where the auxiliary type-F1 peaks (epoch 32, `val_aux_type_f1=0.6949`) the live β is only 0.12. The cross-modal alignment loss `train_align` is identically 0.0 throughout the run because in fused mode the encoder consumes both sensors in a single forward pass, so there is no separate per-sensor signal block to align — the loss is a no-op in this configuration and would only matter in per-sensor mode. The env temporal-stability loss is small (10⁻⁵ to 10⁻⁷) and the signal intervention-invariance loss is small but nonzero (~10⁻²), indicating the disentanglement constraint is mostly being satisfied by the time training stabilizes.

**Classify-side downstream probe (linear_signal on crl_best_aux_type).** The shipping classify probe trains for 50 epochs. Its `val_pres_f1` peaks at 0.7944 at epoch 30 and then drifts slightly downward; its `val_type_f1` peaks at 0.6702 at epoch 40. The downward drift on presence is consistent with the schedule favoring type discrimination over presence as training continues. The numbers reported by the bundle metadata (`pres_f1=0.7944`, `type_f1=0.6702`) correspond to the best-epoch checkpoints saved during this probe training run.

**Detect-side downstream probe (linear_ztype on crl_best).** The shipping detect probe trains for 50 epochs. Its `val_pres_f1` peaks at 0.8714 at epoch 9 (matching the bundle metadata `pres_f1=0.8714` exactly), then plateaus near that level for the remainder. Its `val_type_f1` peaks at 0.4648 at epoch 37, but the type prediction on this probe is not used by the deployed pipeline — the detect bundle ships only the presence head; the classify pod runs an entirely separate type pipeline.

**Convergence summary.** Both runs converge slowly on `val_ref_elbo` (best epoch 98 and 99 of 100 respectively) and substantially earlier on `val_aux_type_f1` (epoch 35 and 32 respectively). This gap is the reason the dual-checkpoint design exists: `crl_best.pth` selected by ref_elbo gives the best-conditioned latent overall while `crl_best_aux_type.pth` selected by aux-type-F1 gives the latent where the type signal is most accessible to a downstream head. Different downstream probes prefer different source checkpoints, which is what the per-bundle selection rule resolves.

![Figure 1. Per-epoch val_ref_elbo across the top-5 ranked CRL runs, y-axis clipped at 10 to show the early-epoch drop. The classify shipping run (2026-05-03_05-03-14) is the red trace; the detect shipping run is not present in this aggregate plot because its saved-run layout pre-dates the per-epoch CSV convention. All five runs follow the same shape: a sharp drop in the first ~5 epochs followed by slow refinement to a plateau near zero. The other traces (blue, orange, green, purple) are non-shipping runs included for context. Source: saved_crl/analysis/val_ref_elbo_over_epochs_clip10.png.](../../crl-train/saved_crl/analysis/val_ref_elbo_over_epochs_clip10.png)

![Figure 2. Same per-epoch val_ref_elbo data as Figure 1, with the y-axis re-clipped at 2 to resolve the post-epoch-10 plateau region. The convergence differences are now visible: most runs settle near val_ref_elbo ≈ 0.2 by epoch 80, while the prior detect candidate 2026-05-03_05-02-44 (purple) plateaus around 0.6 and was stopped at epoch 51 — visibly worse-converged than the rest. The classify shipping run (red) sits at val_ref_elbo ≈ 0.25 by training end. Source: saved_crl/analysis/val_ref_elbo_over_epochs_clip2.png.](../../crl-train/saved_crl/analysis/val_ref_elbo_over_epochs_clip2.png)

![Figure 3. Per-epoch val_aux_type_f1 across the top-5 ranked CRL runs. The classify shipping run (2026-05-03_05-03-14) is the red trace, which peaks at 0.6949 around epoch 32 and stays flat thereafter — this is the trajectory that drives the crl_best_aux_type.pth checkpoint selection. The other traces are non-shipping runs included for context; their downward drift in the second half of training (visible on the orange and purple lines) shows what the dual-checkpoint discipline protects against if a single late-epoch checkpoint were used. Source: saved_crl/analysis/val_type_f1_over_epochs.png.](../../crl-train/saved_crl/analysis/val_type_f1_over_epochs.png)

## Performance Evaluation

Final test-set performance for each shipping bundle, with confidence intervals omitted because the current evaluation pipeline does not compute them. State this honestly: confidence intervals would need to be bootstrapped from the per-window predictions in each `eval/<probe>/<split>/eval_report.json`, which is a future-work item rather than something this document can quote today.

**Detect-side bundle: `multiscale-vae-2026_05_03_15_26_22-v1`.**

| Metric | Value | Source |
|---|---:|---|
| Validation pres_f1 (best probe-epoch) | 0.8714 | `report.json` probes table; `detect-default/meta.json` |
| Cross-location worst-case pres_f1 (`min_pres_f1`) | 0.8021 | `detect-default/meta.json` |
| Validation pres_acc | 0.8009 | `report.json` probes table |
| Source probe (in saved run) | `linear_ztype__crl_best` | `report.json` `save_dir` field |
| Source CRL checkpoint | `crl_best.pth` | bundle source-probe `ckpt_name` |

The detect bundle cleared the side's `pres_f1 ≥ 0.80` promotion floor on both the primary metric and the tie-breaker. The prior candidate (`multiscale-vae-2026_05_03_05_02_44-v1`, `pres_f1=0.863`, `min_pres_f1=0.592`) was eligible by primary metric but failed the tie-breaker by ~21 percentage points on cross-location worst-case, which is why the new bundle was promoted.

**Classify-side bundle: `multiscale-disentangled-2026_05_03_05_03_14-linear_signal-v1`.**

| Metric | Value | Source |
|---|---:|---|
| Validation type_f1 (best probe-epoch) | 0.6702 | `report.json` probes table; `classify-default/meta.json` |
| Cross-location worst-case type_f1 (`min_type_f1`) | 0.4365 | `classify-default/meta.json` |
| Validation type_acc | 0.7603 | `report.json` probes table |
| Validation pres_f1 (collateral, not used in deployment) | 0.7944 | `report.json` probes table |
| Source probe (in saved run) | `linear_signal__crl_best_aux_type` | `report.json` `save_dir` field |
| Source CRL checkpoint | `crl_best_aux_type.pth` | bundle source-probe `ckpt_name` |

The classify bundle cleared the side's `min_type_f1 ≥ 0.40` promotion floor and won the tie-breaker inside an ε=0.01 tie band against two sibling probes from the same source run (see Ablations below). Its headline `type_f1=0.6702` clears the project's capstone target of `type_f1 ≥ 0.65`, but its `min_type_f1=0.4365` does not yet clear the parallel cross-location target of `min_type_f1 ≥ 0.50`; the binding constraint going forward is worst-location F1, not the headline metric.

**Per-class type performance.** On the full evaluation split (91,325 windows aggregated across all source datasets) the shipping classify probe achieves `macro_f1=0.6069` and `accuracy=0.7138`. The 4×4 confusion matrix on the full split is reproduced below, with rows = true class and columns = predicted class, ordered (pedestrian, light, medium, heavy).

|        | pred pedestrian | pred light | pred medium | pred heavy |
|---|---:|---:|---:|---:|
| **true pedestrian** | 3,264 | 988 | 396 | 1,434 |
| **true light** | 361 | 3,083 | 883 | 796 |
| **true medium** | 3,265 | 1,500 | 31,941 | 3,863 |
| **true heavy** | 615 | 1,887 | 3,495 | 10,312 |

Common error modes the matrix exposes:

- The dominant classes (medium and heavy) are handled well — diagonal counts of 31,941/41,569 medium and 10,312/16,309 heavy correspond to per-class recall around 0.77 and 0.63 respectively.
- The two minority classes (pedestrian and light, each ~6% of training) lose accuracy primarily by being misclassified into the dominant classes — pedestrian → heavy is the largest single off-diagonal cell for pedestrian (1,434 windows), and the light → medium / light → heavy errors together account for over 1,600 windows. The aux-type focal-CE rebalancing on the detect side helps but is disabled on the classify side (`use_focal_type=false` per the disentangled run's config), which is one reason the minority-class F1 stays low on this bundle.
- Vehicle-type is genuinely harder than presence at one-second windows because some classes (medium vs. heavy especially) produce overlapping spectral signatures at this window length. Distinguishing them more reliably probably requires a longer window or additional sensor modalities, not a different head architecture.

![Figure 4. Best downstream val_type_f1 per run, top 5 by score. The classify shipping run (2026-05-03_05-03-14) is the red bar; the dashed red line at 0.65 is the project's capstone ship threshold for headline type_f1. The current shipping bundle sits right at the threshold (bundle metadata reports 0.6702, the leaderboard's per-probe number is 0.646), so the headline-F1 target is essentially met; the binding constraint going forward is the cross-location min_type_f1 = 0.4365 vs. the 0.50 worst-location target. The remaining bars are non-shipping runs included for context. Source: saved_crl/analysis/best_f1_by_run.png.](../../crl-train/saved_crl/analysis/best_f1_by_run.png)

## Ablation Studies

Two ablations are treated in depth below; the remainder are listed without depth either because they have no matched-pair data in the current run set or because the relevant baseline is still in progress on the training server.

### In-depth #1: training mode (VAE vs. disentangled)

This is the live shipping decision. The detect side ships VAE because VAE wins on `pres_f1`; the classify side ships disentangled because disentangled wins on `type_f1`. The cleanest cross-comparison comes from comparing the best probe of each kind across the two shipping runs (multiscale frontend in both, ID-split data in both, same optimizer, same batch size, 100 epochs in both).

| Probe | VAE run (detect-side) | Disentangled run (classify-side) |
|---|---|---|
| Best `val_pres_f1` across all 6 probes | **0.8742** (mlp_ztype__crl_best_aux_type) | 0.8438 (linear_signal__crl_best) |
| Best `val_type_f1` across all 6 probes | 0.6551 (linear_fullz__crl_best_aux_type) | **0.6729** (mlp_signal__crl_best_aux_type) |
| Best `val_aux_type_f1` during CRL training | 0.6768 (epoch 35) | **0.6949** (epoch 32) |
| Best `val_ref_elbo` | **0.1926** (epoch 98) | 0.2524 (epoch 99) |

Three observations from this table.

First, **VAE wins presence by ~3 percentage points and disentangled wins type by ~1.8 percentage points**. Neither training mode dominates the other across both objectives, which is the empirical observation that motivates shipping two encoders. A single shared encoder trained for either objective alone would lose roughly 2-3 ppt on the other objective, and there's no training-mode choice in the matrix that wins both columns.

Second, **`val_ref_elbo` favors VAE substantially (0.1926 vs. 0.2524)**. The disentangled run carries three additional loss terms whose value at convergence raises the total objective relative to a pure ELBO, but the comparison is still apples-to-apples because `val_ref_elbo` is computed at β=1 semantics with reconstruction + raw_KL only, ignoring the disentanglement terms. The disentangled mode genuinely sacrifices reconstruction fidelity in exchange for a more linearly-separable signal block, and that's exactly the trade the design is intended to make.

Third, the disentangled run's downstream probes that read the `crl_best_aux_type.pth` checkpoint show a noticeable drop in pres_f1 (~0.79) compared to the same run's probes on `crl_best.pth` (~0.84). This is the dual-checkpoint design at work in the opposite direction from usual: the aux-type checkpoint over-allocates the latent to type discrimination at the cost of presence, so a probe that reads it loses presence accuracy. The classify bundle deliberately ships the type-favoring checkpoint because the bundle is the type pipeline, not the presence pipeline.

### In-depth #2: probe head choice on the disentangled run (linear_signal vs. mlp_signal vs. linear_fullz)

All six classify-side probes were trained on the disentangled run's frozen CRL backbone with identical data, identical 50-epoch protocol, and identical optimizer. The promoted-bundle catalog at `inference-engine/classify-bundles/README.md` summarizes the catalog entries derived from these probes:

| Bundle | Probe head | Reads | type_f1 | min_type_f1 |
|---|---|---|---:|---:|
| `multiscale-disentangled-2026_05_03_05_03_14-linear_signal-v1` (shipping) | `Linear(12, 4)` | z[0:12] | 0.6702 | **0.4365** |
| `multiscale-disentangled-2026_05_03_05_03_14-linear_fullz-v1` | `Linear(24, 4)` | z[0:24] | 0.6704 | 0.4337 |
| `multiscale-disentangled-2026_05_03_05_03_14-mlp_signal-v1` | `Linear(12, 32) → ReLU → Linear(32, 4)` | z[0:12] | **0.6729** | 0.4222 |

All three sit inside the side's ε=0.01 tie band on the primary metric (max-min spread is 0.0027 across the three), so the tie-breaker `min_type_f1` decides — and `linear_signal` wins it by ~0.003 over `linear_fullz` and ~0.014 over `mlp_signal`. The honest reading of this table:

- **`mlp_signal` has the highest headline `type_f1` but the lowest `min_type_f1`**, meaning it's the best on average but the most fragile across locations. The selection rule trades ~0.003 of headline F1 for ~0.014 of worst-location F1, on the design judgment that worst-case is more important to operational reliability than peak-case.
- **`linear_fullz` reading the full 24-dim latent does not meaningfully outperform `linear_signal` reading just the 12-dim signal block**, which is direct empirical evidence that the disentanglement constraint is doing what it was designed to do: the type signal really is concentrated in z[0:12], and the additional 12 env dims contribute essentially nothing on average and slightly hurt worst-case.
- **`linear_signal` is also the simplest deployed model** of the three (`Linear(12, 4)` is 52 parameters), which is a small but real engineering bonus that the selection rule does not consider but that the team values for traceability.

### Listed without in-depth treatment

These ablation axes either have no matched-pair data in the current run set or the relevant counterpart run is still in progress.

- **Frontend (multiscale vs. Morlet)**: a `morlet_per_sensor` baseline run is still in progress on the training server and is expected to complete shortly. Once the run lands and `compare_ablations.py` finds a matched-pair entry between the current shipping multiscale runs and the new Morlet baseline, this section will be promoted to in-depth treatment in a subsequent revision of this document. The earlier multiscale-vs-Morlet head-to-head documented in `saved_crl/slides-figs/model_basics.md` is on an older run generation (`v3_lowfreq`, d_z=24) and is not a clean apples-to-apples comparison against the current shipping config.
- **Frontend Morlet phase channel (`morlet_use_phase` on/off)**: runs exist but `compare_ablations.py` reports no matched pairs.
- **Prior type (standard / learned / conditional)**: only `prior_type=standard` runs exist in the current set; no matched pairs.
- **Stage-2 fine-tune (`stage2` on/off)**: no matched pairs.
- **Morlet learnable w0**: no matched pairs.
- **Auxiliary head ablation (with/without `aux_pres` or `aux_type`)**: not run as a controlled ablation in the current set; both shipping runs train with all aux heads enabled.

The current `saved_crl/analysis/{ablations,leaderboard,cross_location}.md` files include the classify shipping run (`2026-05-03_05-03-14`) but not the detect shipping run (`2026-05-03_15-26-22`). Regenerating the three tables is a one-command operation per file (`python compare_runs.py`, `python compare_ablations.py`, `python compare_cross_location.py`); the revision that promotes the frontend ablation to in-depth treatment will refresh those files and bring the detect run into the leaderboard at the same time.

## References

[1] PyTorch, "torch.nn," PyTorch Documentation. Available: https://pytorch.org/docs/stable/nn.html

[2] D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in Proc. 2nd Int. Conf. Learning Representations (ICLR), 2014. Available: https://arxiv.org/abs/1312.6114

[3] I. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," in Proc. 5th Int. Conf. Learning Representations (ICLR), 2017.

[4] S. Liu et al., "FOCAL: Contrastive learning for multimodal time-series sensing signals in factorized orthogonal latent space," in Proc. 37th Conf. Neural Information Processing Systems (NeurIPS), 2023. Available: https://arxiv.org/abs/2310.20071

[5] A. Vaswani et al., "Attention Is All You Need," in Proc. 31st Conf. Neural Information Processing Systems (NeurIPS), 2017. Available: https://arxiv.org/abs/1706.03762

[6] H. Kim and A. Mnih, "Disentangling by Factorising," in Proc. 35th Int. Conf. Machine Learning (ICML), 2018. Available: https://arxiv.org/abs/1802.05983

[7] T. Lin et al., "Focal Loss for Dense Object Detection," in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017. Available: https://arxiv.org/abs/1708.02002

## Reflection

Three things stood out about producing this Phase 4 training evaluation.

First, **the dual-checkpoint discipline (val_ref_elbo for early-stop versus val_aux_type_f1 for downstream-proxy) was non-obvious early in the project but turned out to be load-bearing**. Using the live training loss for early-stop would have produced ranking instability across the β-annealing schedule, because the same run state has different total-loss values at different live β. Decoupling the early-stop metric from the live β by reporting `val_ref_elbo` at fixed β=1 semantics is the kind of evaluation-protocol decision that has no immediate visible payoff but prevents a class of silent comparison errors that would otherwise show up in the ablation table. The same care in metric design is what made the `min_type_f1` tie-breaker possible: a metric that can rank candidate bundles consistently has to be defined to be comparable across runs first.

Second, **the selection-rule framework (primary metric, tie-breaker, promotion floor, ε tie band) was added because eyeballing leaderboards for "the best one" produced inconsistent shipping decisions earlier in the phase**. Three candidate bundles inside an ε=0.01 of each other on the primary metric is a frequent occurrence in this run set; without a tie-breaker rule, which one ships depends on which order they're listed. The promotion floor exists to prevent shipping a bundle that has a strong headline metric but fails the cross-location worst-case bar — a real failure mode for the prior detect candidate `2026-05-03_05-02-44-v1`, which had `pres_f1=0.863` (good) but `min_pres_f1=0.592` (would have shipped a model that performed catastrophically badly on the FOCAL location). Codifying the rules in `inference-engine/{detect,classify}-bundles/README.md` makes a promotion reproducible from `--promote-default` rather than from a meeting.

Third, **splitting the bundle pipeline per pod is what made the training-mode comparison concrete**. As long as both pods shared a single bundle, the question "VAE or disentangled?" was a forced choice that lost on whichever objective wasn't favored. Once the pipeline split into `detect-bundles/` and `classify-bundles/` with independent default symlinks, the question became "which is better at presence?" and "which is better at type?" — two cleanly answerable questions whose answers differ. The cost is duplicated inference work per window (the same window gets encoded twice, by two different encoders), and we accept that cost on the basis of the F1 numbers in the in-depth #1 ablation table above.

## Appendix

### Glossary of evaluation-specific terms

- **Matched pair**: two saved runs that differ in exactly one configuration variable (e.g., training mode), with all other configuration held fixed. The basic unit of an ablation comparison.
- **F1 macro vs. micro**: macro-F1 is the unweighted average of per-class F1 scores (every class counts equally); micro-F1 weights by class frequency. All `type_f1` numbers in this document are macro-F1.
- **Posterior collapse**: a failure mode of variational training where the latent collapses to the prior (the encoder ignores the input and the KL goes to zero). The β-annealing `kl_floor` exists to detect and counteract this.
- **β-annealing**: a training-time schedule that varies the KL-divergence weight β as a function of training progress and observed KL. See `vae_mode.py` for the exact rule.
- **Tie band (ε=0.01)**: the band around the primary-metric leader inside which candidates are considered statistically indistinguishable on the primary metric, so the tie-breaker decides. ε=0.01 was chosen as a small fraction of the typical between-bundle spread.
- **Promotion floor**: a per-side bundle metric below which a bundle is not eligible to be promoted to the default symlink, regardless of how well it scores on the primary metric.
- **Primary metric / tie-breaker**: the per-side bundle selection rule consists of a primary metric (highest wins) and a tie-breaker (highest wins inside the ε tie band).
- **Source run / source probe**: the saved CRL training run and the specific downstream probe within that run from which a deployed bundle was exported.

### Pointers to live analysis

The `crl-train` repository contains live analysis scripts that regenerate aggregate tables from the current run set:

- `compare_runs.py` — leaderboard CSV and markdown, sortable by any metric column, output to `saved_crl/analysis/leaderboard.{csv,md}`.
- `compare_ablations.py` — pairwise ablation table over the configurable axes (frontend_type, training_mode, prior_type, etc.), output to `saved_crl/analysis/ablations.md`.
- `compare_cross_location.py` — per-location type_f1 matrix and heatmap, output to `saved_crl/analysis/cross_location.{csv,md}` plus `cross_location_heatmap.png`.
- `plot_run.py` — per-run training-curve plots from `crl_metrics.csv` and `downstream_metrics.csv`.
- `plot_aggregate.py` — overlay plots across all runs (val_type_f1 over epochs, val_ref_elbo over epochs, best F1 per run, complexity vs. F1).
- `per_vehicle_confusion.py` — per-vehicle and per-(vehicle, sensor-node) confusion matrices for a given downstream probe.

As of this writing the `saved_crl/analysis/` markdown files include the classify shipping run but not the detect shipping run; refreshing them is a one-line invocation per file.

## Changes To Previous Deliverables

This is the first formal Model Training Evaluation document for the project; Phase 3 had no separate document of this kind. The same four substantive changes documented in the Phase 4 Model Documentation also affect the evaluation picture, summarized here from the evaluation-protocol angle.

**1. Architecture pivot.** The eight Phase 3 baseline architectures (1D CNN, two 2D CNNs, LSTM, miniROCKET, InceptionTime, TCN, BiGRU) are no longer evaluated. The Phase 4 system evaluates two CRL configurations (multiscale-VAE for detect, multiscale-disentangled for classify), independently selected via per-side selection rules. The motivation was the data-leak discovery during the Phase 3 → Phase 4 transition; the original baseline performance had been driven by leakage between adjacent windows, and once the leak was removed the baselines collapsed to performance levels that did not meet the project's operational thresholds.

**2. Dataset construction changes.** The dataset was reconstructed to fix the data leak, and the new ID-split schema (`crl-train/docs/superpowers/specs/2026-04-25-id-split-schema-design.md`) partitions by vehicle/scene identity rather than by file placement. Both shipping runs use this schema (`use_id_split=true`). The split is a 70/15/15 train/val/test partition with class balance verified across all three partitions before training.

**3. Evaluation protocol shift.** Phase 3 reported macro-F1, MCC, ROC-AUC, latency, and false alarm rate across separate detection / category / instance modes. Phase 4 reports `val_pres_f1`, `val_type_f1`, `val_aux_type_f1`, and `val_ref_elbo` from the CRL training pipeline, plus per-class confusion and cross-location worst-case metrics (`min_pres_f1`, `min_type_f1`). Bundle promotion is governed by formal selection rules with primary metrics, tie-breakers, and promotion floors. The instance-identification mode from Phase 3 is no longer evaluated; the architecture is not designed for that task.

**4. Inference engine changes since Phase 3.** Beyond the signal-processing-pod removal already documented in the Phase 3 deliverable, the bundle pipeline has been split into per-pod artifacts (`detect-bundles/` and `classify-bundles/`) with independent default symlinks repointed by `--promote-default`. The legacy single-bundle `CRL_BUNDLE` and `CRL_RUN_DIR` environment variables now produce a hard error in `scripts/build_containers.sh` to prevent silent fallthrough. The two pods now carry independent encoders trained for different objectives and the classify pod re-encodes the inbound waveform rather than reusing the detect pod's z. Per-pod selection rules and TorchScript bundle catalogs replace the previous single ad-hoc model-loading path.
