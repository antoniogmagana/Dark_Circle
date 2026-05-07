# Techniques for Tactical Edge Data Support to the Army Research Laboratory's Live, Virtual, and Constructive (LVC) Toolkit

## Model Documentation (Phase 4)

**Artificial Intelligence (AI) Technician Capstone Group 5**

Students: Brandon Taylor, Antonio Magana, Larry Parrotte, John Tomaselli — Carnegie Mellon University

Mentors: Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Mr. Henry Reimert — DEVCOM Army Research Laboratory

*Author's note: This document, like other Phase 4 deliverables, will be provided to the private-sector partner of this project's Army Research Laboratory (ARL) sponsors, the Greystones Group. As such, we have taken special care to be detailed and clear so that a reader who was not part of model development can build, deploy, and evaluate the system from the cited source files alone.*

---

## Overview

The Phase 4 production model is a **causal representation learning (CRL) pipeline**: a small transformer-based encoder that maps a one-second window of acoustic and seismic data to a compact latent vector, with two downstream heads that read that latent to answer (1) is a vehicle present? and (2) what type of vehicle is it? The system ships **two** independently-trained CRL configurations rather than one. The detect-side configuration is optimized for presence detection; the classify-side configuration is optimized for vehicle-type classification. The two configurations live in two separate inference pods and each carries its own copy of the encoder.

The motivation for splitting the system this way is empirical: the choice of training objective (variational vs. disentangled) trades presence performance against type performance, and a single shared encoder underperforms two specialized encoders on both objectives. Rather than ship a single compromise encoder, the system ships both and lets each pod use the encoder that scored best on its own task. The CRL approach itself (representation learning over a 1-second window with downstream linear probes) was selected because its latents do not depend on memorizing local temporal context, which makes the model robust to the kinds of spectral and temporal variation that appear when the same vehicle is recorded at different locations or sensor nodes.

## Specifications

The shipping system is two CRL bundles, one per inference pod, each backed by a different saved training run. Both bundles share the same backbone topology — a per-sensor multiscale convolutional frontend, a transformer encoder, and a latent space — but differ in latent dimensionality, latent partition, training objective, and the downstream heads carried in the bundle.

### Shared backbone components

**Frontend** (`crl_vehicle/models/frontend.py:10-66`). Each sensor channel is processed by an independent `MultiScale1DFrontend`, which is a stack of parallel learned 1-D convolutions at multiple receptive fields. The audio frontend uses four kernel sizes [9, 19, 39, 159]; the seismic frontend uses three [9, 19, 39]. Each branch is followed by GroupNorm and a GELU activation, adaptively pooled to 32 output tokens, then stacked and projected through a 1×1 convolution to a common embedding width of 64 channels. The branches with different kernel sizes act as a learned analogue to a multi-resolution filter bank, capturing both short-burst and longer-period structure in the same window.

**Fusion** (`training/trainer.py:175-186`). The two per-sensor frontend outputs (each shaped (B, 64, 32)) are concatenated along the time axis to form a single (B, 64, 64) tensor. This early fusion lets the downstream encoder attend across modalities directly rather than treating audio and seismic as separate streams.

**Encoder** (`crl_vehicle/models/encoder_decoder.py:7-46`). A 2-layer transformer with `d_model=64`, 4 attention heads, feed-forward width 128, pre-LayerNorm, no dropout. The encoder consumes the fused 64-token sequence and emits a sequence-mean-pooled latent vector along with two latent heads (one for the posterior mean μ, one for log-variance log σ²). At inference time the latent z is taken to be μ.

**Decoder** (`crl_vehicle/models/encoder_decoder.py:49-69`). Used only during pretraining. A small MLP that reconstructs the fused frontend tensor from z: `Linear(d_z → 64) → GELU → Linear(64 → 4096)`, reshaped to (B, 64, 64). The reconstruction loss is what gives the variational training mode its ELBO objective.

### Detect bundle: `multiscale-vae-2026_05_03_15_26_22-v1`

Backed by the saved run at `crl-train/saved_crl/runs/multiscale/vae/2026-05-03_15-26-22/`.

- **Training mode**: VAE (variational autoencoder) with β-annealing. The latent space is a `CausalLatentSpace` with `d_z=32`, partitioned into named blocks D_PRES=4, D_TYPE=6, D_PROX=3, D_ENV=6, and D_FREE=5 dims (`crl_vehicle/models/latent.py:7-33`).
- **Auxiliary heads during pretraining**: a presence head (`Linear(D_PRES, 1)`), a vehicle-type head (`Linear(D_TYPE, 4)`), and an intervention head on z_env. These supervise the latent so that named blocks carry the information their names imply (`training/trainer.py:111-112`).
- **Exported artifact** in the bundle: `encoder_fused.ts` only — no type head. The encoder maps `(x_audio, x_seismic) → (z[B, 32], pres_logit[B, 1])`. The presence head is fused into the encoder at export time so the detect pod can answer "is a vehicle present?" without loading a separate head module.
- **Bundle metadata** (`inference-engine/detect-bundles/detect-default/meta.json`): `frontend_type=multiscale`, `mode=fused`, `z_dim=32`, `presence_threshold=0.5`, audio at 16 kHz, seismic at 100 Hz, both with one-second windows.

### Classify bundle: `multiscale-disentangled-2026_05_03_05_03_14-linear_signal-v1`

Backed by the saved run at `crl-train/saved_crl/runs/multiscale/disentangled/2026-05-03_05-03-14/`.

- **Training mode**: disentangled VAE. The latent space is a `SplitLatentSpace` with `d_z=24` partitioned into a `signal` block (`z[0:12]`, the first 12 dims) and an `env` block (`z[12:24]`, the remaining 12 dims) (`crl_vehicle/models/latent.py:55-69`). The objective adds three terms to the ELBO: cross-modal alignment between the audio and seismic signal blocks, env temporal stability across nearby windows, and signal intervention-invariance under synthetic noise augmentations (`crl_vehicle/training_modes/disentangled_mode.py:45-82`).
- **Auxiliary heads during pretraining**: `LinearPresenceHead(d_in=12)` and `LinearTypeHead(d_in=12, n_classes=4)`, both reading the signal block. Owned by the training mode rather than the model so they are sized to `d_signal` rather than to a fixed presence/type sub-block.
- **Downstream probe** (the type head shipped in the bundle): `LinearTypeHead(d_in=12, n_classes=4)` — a single linear layer reading `z[0:12]`. This is the `linear_signal` probe defined in `crl_vehicle/models/heads.py` (see also the table at `crl_vehicle/models/README.md:146`).
- **Exported artifacts** in the bundle: `encoder_fused.ts` (a copy of the disentangled-run encoder, separate from the detect-side encoder) plus `type_head_fused.ts`.
- **Bundle metadata** (`inference-engine/classify-bundles/classify-default/meta.json`): `frontend_type=multiscale`, `mode=fused`, `z_dim=24`, `probe_mode=linear_signal`, `class_names=["pedestrian", "light", "medium", "heavy"]`.

### Inference data flow

Per ROS2 sensor topic, the ingestor pod assembles a one-second window with DC removal and publishes a `SensorData` protobuf to the NATS `sensor.data` subject. The `infer-detect` pod consumes that subject and runs the **detect bundle's** encoder, emitting the latent z and the presence logit. If the presence logit clears the bundle's threshold the result is published to `detection.result`. The `infer-classify` pod subscribes to `detection.result` and **re-encodes the original waveform with the classify bundle's encoder** — it does not reuse the detect-side z, because the two bundles' encoders were trained for different objectives. The classify pod then runs the linear_signal type head on its own z and publishes the predicted class to `classification.result`. The egress pod merges the two streams onto a single `/inference_result` ROS2 topic, publishing exactly one message per window. (`inference-engine/src/infer_classify/main.py:85-94` for the re-encode behavior; `inference-engine/classify-bundles/README.md:18-21` for the design rationale.)

The framework supports several configurable axes that the shipping bundles do not exercise. Frontends include morlet_per_sensor, morlet_fused, and morlet_learnable wavelet-based variants. Training modes include a contrastive mode in addition to VAE and disentangled. Classify-side probes include linear_ztype, mlp_ztype, mlp_signal, and linear_fullz. The two bundle catalogs at `inference-engine/detect-bundles/README.md` and `inference-engine/classify-bundles/README.md` document the naming convention, the selection rules used to promote a bundle to default, and the current catalog entries. Adding a new bundle is a matter of training a CRL run, exporting it with `crl-train/export_for_inference.py --bundle-kind {detect,classify} --bundle-name <name>`, and optionally running `--promote-default` to re-evaluate which bundle wins under the selection rules.

![Figure 1. Dual-bundle CRL inference data flow. The ingestor pod publishes one-second sensor windows to NATS sensor.data; both inference pods subscribe and run their own bundle's encoder. The detect pod's encoder (VAE, d_z=32) emits a presence logit, and only positive predictions drive the classify pod, which re-encodes the same waveform with its own encoder (disentangled, d_z=24, d_signal=12) and runs the linear_signal type head on z[0:12]. The egress pod merges both result streams onto a single ROS2 /inference_result topic. Source: saved_crl/analysis/dual_bundle_dataflow.png.](../../crl-train/saved_crl/analysis/dual_bundle_dataflow.png)

## Model Run

Both shipping CRL configurations were trained on a dedicated Supermicro IoT Superserver equipped with an Intel Xeon Platinum 8581V processor, an NVIDIA H100 NVL graphics card, and 1 terabyte of RAM. The training pipeline is implemented in PyTorch with nearly all compute operations routed to the GPU. Training data is drawn directly from a PostgreSQL database rather than from flat files, with batch retrieval typically taking under 20 ms per batch. Eight CPU background workers stage data batches in RAM ahead of GPU consumption to minimize I/O bottlenecks.

Each CRL pretraining run takes approximately eight and a half hours: the detect-side run completed 100 epochs in 512.7 minutes, and the classify-side run completed 100 epochs in 549.3 minutes (`report.json` `crl_elapsed_min` field for each run). After CRL pretraining each run produces a small set of downstream probes — six probes for the detect run, six for the classify run — each of which trains in a few minutes against the frozen CRL backbone.

The Supermicro server represents a significant capital investment of approximately $48,000 for the H100 NVL, Xeon Platinum, RAM, and storage combined. This cost is justified by the need for rapid retraining and high-throughput experimentation during architecture development. For deployed inference the LVC Toolkit's intended purpose requires edge operation, and the trained TorchScript bundles run in a Kubernetes cluster on commodity edge hardware — the inference pods do not need a GPU and can serve a one-second window in well under the real-time deadline. With additional platform adjustments the system could be adapted for deployment on a portable 16 × 10.5 × 4.9 inch edge server with 2× 800 W power supplies, opening the possibility of LVC Toolkit use in moving vehicles or austere field conditions with no network connectivity.

## Training

Both shipping runs share a common training procedure with mode-specific loss components. All values quoted here come from each run's `crl/meta.json` (or, for the detect run which lacks a `crl/` subdirectory, from the `report.json` `crl.config` field).

**Optimization (both runs)**. AdamW optimizer with learning rate 3 × 10⁻⁴, weight decay 1 × 10⁻⁴, batch size 128, eight DataLoader workers. Learning rate is annealed by a cosine schedule from 3 × 10⁻⁴ down to 1 × 10⁻⁴ over 100 epochs. Maximum 100 epochs, early-stop patience of 25 epochs monitoring `val_ref_elbo` (a β-invariant reference ELBO that is comparable across the β-annealing schedule, defined in `crl_vehicle/training_modes/vae_mode.py`). Both shipping runs used the full 100-epoch budget without triggering early stop. Random seed 42.

**Detect-side loss components (VAE)**. ELBO with β-annealing: `β_step=0.02`, `kl_floor=0.01`, `kl_target=0.5`, `recon_min_delta=1 × 10⁻⁴`. The β scalar is increased when reconstruction is improving or when the KL exceeds the target, decreased when the KL falls below the floor (which would indicate posterior collapse), and held otherwise. Three auxiliary losses are added to the ELBO with weights `λ_interv=1.0`, `λ_aux_pres=1.0`, `λ_aux_type=1.0` and `λ_aux_prox=0.1`. The vehicle-type auxiliary loss uses focal cross-entropy with γ=2.0 to compensate for class imbalance.

**Classify-side loss components (disentangled)**. ELBO over the `SplitLatentSpace` plus three disentanglement terms: cross-modal alignment with `λ_align=1.0`, env temporal stability with `λ_stab=0.1`, and signal intervention-invariance with `λ_interv_inv=1.0`. The same three auxiliary heads (presence, type, prox) are attached with the same weights as the detect side, but the type loss uses standard cross-entropy rather than focal CE (`use_focal_type=false` in this run's config). The KL annealing schedule is the same as the detect side.

**Checkpointing (both)**. Each run saves three checkpoints: `crl_best.pth` selected by the lowest `val_ref_elbo` (the β-invariant reference metric), `crl_best_aux_type.pth` selected by the highest `val_aux_type_f1` (a downstream-proxy signal that often peaks earlier than ref_elbo), and `crl_final.pth` from the last epoch. The two-checkpoint design exists because the β-annealing schedule means the live training loss is non-stationary and would not produce a comparable early-stop signal across epochs; ref_elbo is computed at β=1 semantics regardless of the live β so it remains comparable. Which checkpoint a downstream bundle uses is a per-bundle decision driven by the bundle's selection rule (see Evaluation).

**Downstream probe training (classify side)**. After CRL pretraining the classify run trains six small probe heads on the frozen CRL backbone, one for each combination of probe mode (linear_signal, mlp_signal, linear_fullz) and source checkpoint (crl_best, crl_best_aux_type). Each probe is trained for 50 epochs with AdamW at the same learning rate, with separate optimizers for the presence and type heads. The probe selected for shipping (`linear_signal` on `crl_best_aux_type`) is the one that won the classify-side selection rule — see Evaluation.

**Datasets and splits**. Both runs draw from the same combined dataset described in `Dark_Circle/DATASET_CONSTRUCTION.md` and use the ID-split schema defined in `crl-train/docs/superpowers/specs/2026-04-25-id-split-schema-design.md` (`use_id_split=true` in both meta.json files). The full dataset and split protocol is documented in the companion Model Training Evaluation document; in brief, the data combines the M3NVC, MOD_vehicle IOBT, and MOD_vehicle FOCAL collections, with synthetic intervention-noise augmentations applied during training only.

## Evaluation

Each shipping bundle was selected from a catalog of candidate bundles by an explicit selection rule documented in the bundle catalog README. The detect-side rule promotes the bundle with the highest `pres_f1` on the held-out validation split, with `min_pres_f1` (the worst per-location F1) as the tie-breaker inside an ε=0.01 tie band, subject to a `pres_f1 ≥ 0.80` promotion floor. The classify-side rule promotes the bundle with the highest `type_f1`, with `min_type_f1` as the tie-breaker inside the same tie band, subject to a `min_type_f1 ≥ 0.40` promotion floor. The promotion floor exists to prevent shipping a bundle that does not even clear the cross-location worst-case bar, regardless of how well it performs on the headline metric.

### Detect-side performance

The shipping detect bundle reports `pres_f1=0.8714` and `min_pres_f1=0.8021` (`detect-bundles/detect-default/meta.json`). It was exported from the `linear_ztype__crl_best` probe of the 2026-05-03_15-26-22 VAE run, which is the probe whose val_pres_f1 matches the bundle metadata exactly (`report.json` probes table). Several other probes on the same run achieved higher peak val_pres_f1 (mlp_ztype__crl_best_aux_type reached 0.8742) but did not clear the tie-breaker on min_pres_f1; the bundle promotion follows the side's selection rule rather than the headline-only ranking.

The bundle clears the side's `pres_f1 ≥ 0.80` promotion floor on both the primary metric and the tie-breaker. The min_pres_f1 of 0.802 sits just over the floor and reflects worst-case behavior on the FOCAL location, which has a different dataset distribution than the training-dominant M3NVC collection.

### Classify-side performance

The shipping classify bundle reports `type_f1=0.6702` and `min_type_f1=0.4365` (`classify-bundles/classify-default/meta.json`), backed by the `linear_signal__crl_best_aux_type` probe of the 2026-05-03_05-03-14 disentangled run. Three sibling bundles on the same source run all sit inside the ε=0.01 tie band on the primary metric, so the tie-breaker decided.

| Bundle | Probe | type_f1 | min_type_f1 | Notes |
|---|---|---:|---:|---|
| `multiscale-disentangled-2026_05_03_05_03_14-linear_signal-v1` | `linear_signal` | 0.670 | **0.437** | Current `classify-default`. Wins on the tie-breaker. |
| `multiscale-disentangled-2026_05_03_05_03_14-linear_fullz-v1` | `linear_fullz` | 0.670 | 0.434 | Inside the tie band; second on the tie-breaker. |
| `multiscale-disentangled-2026_05_03_05_03_14-mlp_signal-v1` | `mlp_signal` | **0.673** | 0.422 | Highest headline `type_f1` but lowest `min_type_f1`. |

The shipping bundle's headline `type_f1=0.6702` clears the project's capstone target of `type_f1 ≥ 0.65`, but its `min_type_f1=0.4365` does not yet clear the parallel cross-location target of `min_type_f1 ≥ 0.50`. The binding constraint going forward is worst-location F1, not headline F1. The selection rule trades a 0.003 advantage in headline `type_f1` (held by `mlp_signal`) for a 0.014 advantage in worst-location F1 (held by `linear_signal`), prioritizing cross-location robustness over peak performance — which aligns with the binding constraint.

### Per-class type performance

On the full evaluation split (91,325 windows) the shipping classify probe achieves macro F1 = 0.607 and accuracy = 0.714. The 4×4 confusion matrix below shows the per-class behavior: medium recall is 79% and heavy recall is 63%, while pedestrian (54%) and light (60%) carry most of the per-class F1 loss. The dominant pedestrian failure mode is misclassification as heavy (24% of true pedestrians); for light the failures are split between medium (17%) and heavy (16%). The `min_type_f1=0.437` reported in the bundle metadata is the cross-location worst-case summary that the selection rule's tie-breaker is written against.

![Figure 2. Vehicle-type confusion matrix for the classify shipping bundle (multiscale-disentangled-2026_05_03_05_03_14-linear_signal-v1) on the full evaluation split (n=91,325 windows). Cells show count and row-percentage; rows = true class, columns = predicted class. Cell color encodes per-row recall on a 0–1 scale. Source: saved_crl/analysis/type_confusion_full_split_2026-05-03_05-03-14.png, rendered from the report.json eval entry.](../../crl-train/saved_crl/analysis/type_confusion_full_split_2026-05-03_05-03-14.png)

### A note on analysis-file coverage

The aggregated analysis tables at `saved_crl/analysis/{leaderboard,cross_location,ablations}.md` currently include the classify shipping run (`2026-05-03_05-03-14`) but not the detect shipping run (`2026-05-03_15-26-22`). The numbers quoted in this section come directly from each bundle's `meta.json` and the `report.json` of its source run, so they are correct, but the aggregated tables should be regenerated to bring the detect run into the leaderboard before they are referenced as canonical.

## References

[1] PyTorch, "torch.nn," PyTorch Documentation. Available: https://pytorch.org/docs/stable/nn.html

[2] D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in Proc. 2nd Int. Conf. Learning Representations (ICLR), 2014. Available: https://arxiv.org/abs/1312.6114

[3] I. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," in Proc. 5th Int. Conf. Learning Representations (ICLR), 2017.

[4] S. Liu et al., "FOCAL: Contrastive learning for multimodal time-series sensing signals in factorized orthogonal latent space," in Proc. 37th Conf. Neural Information Processing Systems (NeurIPS), 2023. Available: https://arxiv.org/abs/2310.20071

[5] A. Vaswani et al., "Attention Is All You Need," in Proc. 31st Conf. Neural Information Processing Systems (NeurIPS), 2017. Available: https://arxiv.org/abs/1706.03762

[6] Raspberry Shake, "Raspberry Shake basic concepts." Available: https://raspberryshake.org

[7] restoreml, "m3n-vc: Multi-modality multi-node vehicle classification dataset," GitHub repository. Available: https://github.com/restoreml/m3n-vc

[8] J. O. Smith, "Spectral audio signal processing," Center for Computer Research in Music and Acoustics, Stanford University. Available: https://ccrma.stanford.edu/~jos/sasp/

## Reflection

Three things stood out about producing this Phase 4 model documentation.

First, the dual-bundle architecture is the most consequential design choice the project made in this phase, and its justification is empirical rather than theoretical. Optimizing a single shared latent for both presence and type pulled the latent in different directions enough that a compromise encoder underperformed two specialized encoders on both objectives. The cost of the dual-bundle design is duplicated inference work — the same window passes through two encoders — and we accept that cost on the basis of the F1 gains on both presence and type. The lesson is that "one model per task" can outperform "one model for the whole problem" even when the tasks share most of their input representation, and that this kind of decision is decided by running both versions and measuring rather than by reasoning about what the right factorization should be.

Second, the selection-rule framework on the bundle catalog (primary metric, tie-breaker, promotion floor, ε tie band) made shipping decisions reproducible. With several candidate bundles often clustering inside an ε of each other on the headline metric, the rule resolves which one ships without depending on which order they happen to be listed or who is in the room when the decision is made. Encoding the rule in `inference-engine/{detect,classify}-bundles/README.md` and implementing it in `--promote-default` means a bundle promotion is now an audit-able operation rather than a judgment call.

Third, much of the documentation work for this phase was code-and-artifact archaeology rather than writing. The shipping bundles encode their provenance back to a saved run, the saved run encodes its config in `meta.json`, the config refers to code in `crl_vehicle/`, and getting numbers right meant tracing every claim back to one of those four levels rather than paraphrasing from memory. The repository's analysis markdown files lagged behind the current run set by a few days at the time of writing, which is the kind of drift the next phase should address by automating the regeneration of those files.

## Glossary

- **AdamW**: Adam optimizer with decoupled weight decay; the optimizer used by both shipping CRL training runs.
- **Auxiliary head**: a small classifier attached to the latent during pretraining that supervises a specific block of the latent (presence, type, intervention) so that the named blocks carry the information their names imply.
- **β-VAE / β-annealing**: a variant of the VAE objective in which the KL-divergence term is multiplied by a scalar β. β-annealing varies β during training to balance reconstruction quality against latent regularization.
- **Bundle**: a self-contained directory of TorchScript artifacts plus a `meta.json` that an inference pod loads at startup. The shipping system uses two bundle catalogs, `detect-bundles/` and `classify-bundles/`, one per pod.
- **CRL (causal representation learning)**: the training-objective family used by this project's shipping models. Latents are trained to capture content rather than memorize timing.
- **D_PRES, D_TYPE, D_PROX, D_ENV, D_FREE**: the named blocks of the `CausalLatentSpace` used by the VAE training mode (sizes 4, 6, 3, 6, 5 in d_z=32). Used by the detect-side bundle.
- **d_signal / z_signal / z_env**: the two-block partition used by the disentangled training mode. `d_signal=12` means the first 12 dims of the latent are the signal block (containing vehicle-type-relevant information) and the remaining dims are the env block.
- **Disentangled (training mode)**: an ELBO-based training mode that adds three losses (cross-modal alignment, env temporal stability, signal intervention-invariance) to encourage the latent to factor into signal and env blocks.
- **ELBO (evidence lower bound)**: the variational objective that combines reconstruction and KL terms. The base objective for both shipping training modes.
- **Frontend**: the per-sensor preprocessing block that converts raw waveform tokens into a fixed-length, fixed-channel feature sequence consumed by the encoder. The shipping bundles use the multiscale (learned convolutional) frontend.
- **Fused mode**: an inference-time mode in which a single encoder consumes both audio and seismic in one forward pass. The alternative, per-sensor mode, runs separate encoders per modality. Both shipping bundles use fused mode.
- **KL floor / KL target**: thresholds in the β-annealing schedule. β decreases when the KL falls below the floor (preventing posterior collapse) and increases when the KL exceeds the target.
- **Min F1 / min_pres_f1 / min_type_f1**: the worst per-location F1 across the locations in the held-out evaluation split. Used as the tie-breaker in the bundle selection rules.
- **MLP probe / linear probe**: a small classifier trained on top of a frozen CRL backbone. Linear probes are a single linear layer; MLP probes have one hidden layer with ReLU. Probes are how downstream tasks (presence, type) are evaluated against a CRL backbone.
- **Presence threshold**: the threshold applied to the presence logit by the detect pod to convert a continuous logit into a binary detection. Set to 0.5 in both shipping bundles.
- **Promotion floor**: a per-side bundle metric below which a bundle is not eligible to be promoted to the default symlink, regardless of how well it scores on the primary metric. `pres_f1 ≥ 0.80` for detect, `min_type_f1 ≥ 0.40` for classify.
- **Reference ELBO (val_ref_elbo)**: the ELBO computed at β=1 semantics regardless of the live β-annealing schedule. Used as the early-stop and `crl_best.pth` selection metric because it is comparable across the non-stationary training schedule.
- **Selection rule**: the formal procedure by which a bundle is promoted to the default symlink. Each side has a primary metric, a tie-breaker, a promotion floor, and an ε=0.01 tie band; see the bundle catalog READMEs.
- **TorchScript bundle**: a compiled-and-serialized PyTorch model that can be loaded by `torch.jit.load` without depending on the original Python source. The format used to ship CRL bundles to the inference pods so the pods do not need a copy of `crl-train` at runtime.

## Changes To Previous Deliverables

Four substantive changes have occurred since the Phase 3 model documentation, listed in order of impact.

**1. Architecture pivot.** The eight baseline architectures documented in Phase 3 (1D CNN, two 2D CNNs, LSTM, miniROCKET, InceptionTime, TCN, BiGRU) have been entirely superseded. During the Phase 3 → Phase 4 transition the team identified data leakage in the dataset construction pipeline; once the leak was removed, the reported strong performance of those baselines collapsed because their results had been driven by leakage between adjacent windows. The architecture pivoted to a CRL-based pipeline whose representations do not depend on memorizing local temporal context. The current shipping system uses **two** CRL bundles — one VAE-trained for detection, one disentangled-trained for classification — optimized independently for the two operational tasks.

**2. Dataset construction changes.** The dataset was reconstructed to fix the data leak that drove the Phase 3 baseline collapse. See `Dark_Circle/DATASET_CONSTRUCTION.md` for the detailed account and the new id-split schema spec at `crl-train/docs/superpowers/specs/2026-04-25-id-split-schema-design.md` for the split protocol. Both shipping CRL runs use the ID-split mode (`use_id_split=true`). Per-modality bit-depth handling (audio 16-bit, seismic and accelerometer 24-bit) has been moved into the inference engine's `sensor-config` ConfigMap so it is part of the deployment configuration rather than per-pod code.

**3. Evaluation protocol shift.** Phase 3 reported macro-F1, MCC, ROC-AUC, latency, and false alarm rate across separate detection / category / instance modes. Phase 4 reports `val_pres_f1`, `val_type_f1`, `val_aux_type_f1`, and `val_ref_elbo` from the CRL training pipeline, plus per-class confusion and cross-location `min_pres_f1` / `min_type_f1`. Bundle promotion is governed by formal selection rules with primary metrics, tie-breakers, and promotion floors documented in the bundle catalog READMEs. The instance-identification mode from Phase 3 is no longer evaluated; the architecture is not designed for that task.

**4. Inference engine changes since Phase 3.** Beyond the signal-processing-pod removal already documented in the Phase 3 deliverable:

- The bundle pipeline has been split into per-pod artifacts. The previous single `crl-bundles/` directory has been replaced by sibling `detect-bundles/` and `classify-bundles/` directories with independent default symlinks. The legacy single-bundle `CRL_BUNDLE` and `CRL_RUN_DIR` environment variables now produce a hard error in `scripts/build_containers.sh:39-50` to prevent silent fallthrough during the transition.
- Sensor discovery moved from prefix-grouping over the ROS2 graph to an operator-curated `expected-sensors.yaml` ConfigMap, with role binding (acoustic, seismic, accel-x/y/z) injected to the ingestor pod as a `SENSOR_ROLE_MAP` JSON env var rather than inferred from message-content suffixes.
- The CRL TorchScript bundle pipeline is now the only production model-loading path; build-time selection happens via `DETECT_BUNDLE` and `CLASSIFY_BUNDLE` environment variables, and the inference pods carry no Python model code at runtime.
- Per-pod default symlinks (`detect-default`, `classify-default`) are repointed by `--promote-default` on the export script, which re-evaluates the catalog against the side-specific selection rules.
- Latency instrumentation (`publish_time` and `latency_seconds` fields) has been added to every inference result message so end-to-end latency can be observed without reconstructing it from log timestamps.
- The egress pod now publishes exactly one message per window — detection-only for negatives, full classification for positives — fixing a Phase 3 bug in which positive predictions were published twice (once from the detect pod, once from the classify pod).
- `TORCH_NUM_THREADS=2` is now set on the inference pods to match the cgroup CPU limit; without this, PyTorch was spawning threads for every host CPU and thrashing under contention with adjacent pods.
