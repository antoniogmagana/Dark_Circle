# CRL-Train: Architecture, Training, and Validation

---

## 1. System Overview

CRL-Train is a Causal Representation Learning (CRL) pipeline for multi-modal vehicle detection. It learns a structured, disentangled latent space using a Variational Autoencoder (VAE) backbone. Causality is enforced via CITRIS-style temporal intervention matching, where the model learns to identify which latent factor has changed between consecutive time steps `t` and `t+n`.

The latent space is semantically partitioned into vehicle *presence* (1D), *type* (4D), *proximity* (1D), and *unstructured noise* (4D).

**Signal flow (per modality, per time step):**
```
(B, C, W) raw window
  → LearnableFilterbank  → (B, K*C, T')  log-power envelope
  → TemporalSSM          → (B, T', d_model)
  → CausalEncoder        → (B, d_z)  [z, mu, log_var]
  → SCM                  → (B, d_z)  z_scm
  → SpectralDecoder      → (B, K, T') x_hat
  → VehicleDetectionHead → (presence_logit, type_logits)
```

---

## 2. Configuration (`crl_vehicle/config.py`)

### `ModalityConfig` (dataclass)
Per-modality signal processing parameters.

| Field | Description |
|---|---|
| `sample_rate` | Target sample rate after resampling |
| `window_size` | Samples per window (= 1 s at target SR) |
| `n_channels` | Number of sensor channels |
| `n_filters` | Number of filterbank bands |
| `filter_len` | FIR kernel length (samples) |
| `envelope_pool`, `envelope_stride` | AvgPool window/stride for envelope extraction |
| `freq_init` | Init strategy: `"log"`, `"seismic"`, `"audio"` |
| `f_min`, `f_max` | Learnable center frequency bounds (Hz) |

**Properties:**
- `t_prime` — temporal steps after envelope pooling: `window_size // envelope_stride` (= 25 for both modalities)
- `filterbank_out_channels` — total feature channels: `n_filters * n_channels`

### `CRLConfig` (dataclass)
Full pipeline config; the single source of truth passed to every component.

**Latent structure:** `d_z = 10` total (`d_z_presence=1`, `d_z_type=4`, `d_z_proximity=1`, `d_z_noise=4`)

**Key training hyperparameters:**

| Parameter | Value | Role |
|---|---|---|
| `beta_start / beta_end` | 0.0 / 4.0 | KL weight annealing |
| `lambda_causal` | 1.0 | SCM consistency weight |
| `lambda_interv` | 2.0 | Intervention identification weight |
| `lambda_disent` | 0.5 | Total correlation penalty weight |
| `lambda_task` | 1.0 | Downstream task weight |
| `lambda_acyclic` | 1.0 | NOTEARS DAG constraint weight |
| `unknown_interv_start_epoch` | 10 | Curriculum: when unknown-interv training begins |
| `unknown_interv_ramp_epochs` | 10 | Curriculum: ramp-up duration (epochs 10–20) |

**Helper constructors:**
- `default_audio_config()` — SR=4000, window=4000, 4 filters, freq range 20–1800 Hz
- `default_seismic_config()` — SR=200, window=200, 4 filters, freq range 2–90 Hz

---

## 3. Data Pipeline (`crl_vehicle/data/`)

### `dataset.py`

#### `_parse_stem(stem, sensor) → tuple | None`
Parses a parquet filename into `(dataset, vehicle, rs_node)`. Returns `None` if the filename does not match the expected pattern. Used during index construction.

#### `_vehicle_to_labels(dataset, vehicle) → (int, bool)`
Maps a `(dataset_name, vehicle_name)` pair to a numeric `vehicle_type` class index and a validity flag. Returns `(-1, False)` for unknown vehicles, which are filtered from training.

#### `collate_single(batch) → dict[str, Tensor]`
Custom collate function for `SensorDataset`. Stacks per-sample dicts into batched tensors. Handles integer, float, and boolean fields uniformly.

#### `collate_pairs(batch) → dict[str, Tensor]`
Collate for `ConsecutivePairDataset`. Stacks `_t` and `_t1` time-step dicts.

---

#### `SensorDataset` (torch.utils.data.Dataset)
Main dataset. Two-pass construction:
1. **Pass 1 `_build_index()`** — groups parquet files by `(dataset, vehicle, rs_node, seg_key)`, loads all files into RAM cache, counts windows per segment.
2. **Pass 2** — resolves vehicle labels, builds the flat index of `(group_key, window_idx, vehicle_type, det_label, audio_seg_id, seismic_seg_id)`.

**`_build_index(parquet_dir, is_train)`**
Walks the parquet directory, groups files by modality/stem, loads each into the RAM cache via `_load_file`, records group window counts. Called once in `__init__`.

**`_load_file(path, stem, sensor, dataset) → dict`**
Reads a single parquet file. Returns `{seg_key: (stem, n_windows)}`. The parquet is parsed by `_df_to_entry` and stored in the cache.

**`_df_to_entry(df, sensor, native_sr) → dict`**
Converts a parquet DataFrame into `{data: Tensor(C, N), present: bool, native_sr: int}`. Handles missing/all-zero columns as `present=False`.

**`_resample(tensor, orig_sr, sensor) → Tensor`**
Resamples a raw signal tensor from `orig_sr` to the modality's target SR using `torchaudio.functional.resample`. Caches results to avoid redundant computation.

**`_get_window(sensor, stem, seg_key, w, interv_idx) → Tensor`**
Core per-sample extraction. Slices window `w` from the cached segment, resamples, applies RMS normalization, and applies a synthetic intervention with 60% probability during training.

**`_zero_window(sensor) → Tensor`**
Returns a zero-filled `(C, W)` tensor for missing modalities. Used when one sensor is absent for a given sample.

**`__getitem__(idx) → dict`**
Returns a sample dict with keys: `x_audio`, `x_seismic`, `audio_avail`, `seismic_avail`, `interv_idx`, `vehicle_type`, `detection_label`, `segment_id`.

---

#### `ConsecutivePairDataset` (torch.utils.data.Dataset)
Wraps `SensorDataset`. At construction, scans the flat index for consecutive window pairs `(i, i+1)` within the same group. Used for CITRIS-style unknown-intervention training.

**`__getitem__(idx) → dict`**
Returns a combined dict with `_t` and `_t1` suffixed keys. Each time step receives an independently sampled intervention, creating the contrastive signal needed for the unknown-intervention classifier.

---

### `transforms.py`

#### `rms_normalize(x, eps=1e-8) → Tensor`
Normalizes a `(C, W)` window to unit RMS per channel. Applied to every window before batching. Using per-window (not dataset-level) statistics preserves transient vehicle amplitude signatures without requiring global statistics.

#### `add_awgn(x, snr_db_range=(10, 30)) → Tensor`
Adds white Gaussian noise at a uniformly sampled SNR in the specified range. Augmentation for robustness against ambient electrical noise.

#### `random_time_shift(x, max_shift_frac=0.1) → Tensor`
Circular shift along the time axis by a random fraction of the window length. Encourages the encoder to be invariant to small temporal offsets.

#### Noise generators (internal)
Each takes `(L,)` length and optional `sample_rate`, returns a `(L,)` mono noise tensor:

| Function | Spectral character | Physical analogue |
|---|---|---|
| `_white_noise` | Flat | Rain, sensor static |
| `_brown_noise` | 1/f² | Wind, low-freq rumble |
| `_pink_noise` | 1/f | Generic ambient |
| `_green_noise` | Band-pass ~500 Hz | Mid-freq ambient |
| `_low_freq_osc` | 2–12 Hz sinusoids | Thunder, distant traffic |
| `_high_freq_chirp` | Linear sweep, upper band | Squeaks, metal contact |
| `_bird_chirps` | Modulated bursts, upper band | Bird calls |

#### `apply_intervention(x, intervention_id, sample_rate) → Tensor`
Injects a synthetic noise source into a `(C, W)` window. `intervention_id=0` is a no-op; IDs 1–7 map to the noise generators above. Noise is scaled to 20% of the signal's RMS. This is the ground-truth intervention used for the known-intervention training path.

**Constant `N_INTERVENTIONS = 7`** — number of distinct synthetic noise types.

---

## 4. Feature Extraction (`crl_vehicle/models/`)

### `filterbank.py`

#### `sinc_bandpass(f_centers, bandwidth, L, sr) → Tensor`
Builds sinc-windowed FIR bandpass kernels of shape `(K, 1, L)`. Each kernel is the difference of two sinc functions (high-cut minus low-cut), windowed by a Hann window, and L2-normalized. The bandwidth is held constant (`f_center / 4` → constant-Q).

#### `_log_spaced_centers(f_min, f_max, K) → Tensor`
Returns `K` log-uniformly spaced frequencies between `f_min` and `f_max` Hz.

#### `_vehicle_seismic_centers(K) → Tensor`
Domain-specific seismic initialization: five bands tuned to vehicle ground-coupling frequencies (2–8, 8–20, 20–40, 40–70, 70–98 Hz).

#### `_vehicle_audio_centers(K) → Tensor`
Domain-specific audio initialization: four bands tuned to engine/tire harmonics (20–150, 150–500, 500–1200, 1200–1800 Hz).

#### `LearnableFilterbank` (nn.Module)
Learnable spectral filterbank. Center frequencies are stored as `log_f_center` (preventing negative frequencies). Bandwidth is fixed. Kernels are rebuilt from `log_f_center` on every forward pass.

**`_build_kernels() → Tensor`**
Reconstructs `(K*C, 1, L)` depthwise conv kernels by exponentiating `log_f_center` and calling `sinc_bandpass`. Called inside `forward`.

**`forward(x: (B,C,W)) → (B, K*C, T')`**
1. Builds current kernels.
2. Applies grouped 1D convolution (one filter bank per channel).
3. Squares output → instantaneous power.
4. AvgPool over `envelope_pool / envelope_stride` → temporal envelope.
5. Log-compresses: `log(power + 1e-6)`.

**`center_frequencies() → Tensor`**
Returns current learnable center frequencies in Hz (for diagnostics/logging).

---

### `ssm.py`

#### `_CausalTransformerSSM` (nn.Module)
Standard Transformer encoder with a lower-triangular causal attention mask. Prevents any time step from attending to future steps. Standard `nn.TransformerEncoder` stack.

**`forward(x: (B,T',d_model)) → (B,T',d_model)`**
Passes input through causal Transformer layers.

#### `_try_mamba_backend(in_channels, config)` → nn.Module | None`
Attempts to import and instantiate a Mamba SSM. Returns `None` if the `mamba_ssm` package is not installed, allowing transparent fallback to the Transformer.

#### `TemporalSSM` (nn.Module)
Public temporal modeling interface.

**`forward(x: (B, K*C, T')) → (B, T', d_model)`**
1. Permutes to `(B, T', K*C)`.
2. `proj_in`: Linear + LayerNorm → `(B, T', d_model)`.
3. Adds learned positional embeddings (up to 256 time steps).
4. Passes through Transformer or Mamba SSM.

---

### `encoder.py`

#### `CausalEncoder` (nn.Module)
Maps SSM output to a structured latent variable `z` via soft attention pooling.

**`forward(x: (B,T',d_model)) → (z, mu, log_var)` each `(B, d_z)`**
1. Computes per-step attention score via `attn_score` linear layer → softmax over `T'`.
2. Weighted sum over `T'` → `(B, d_model)` context vector.
3. Projects to `(B, 2*d_z)` → split into `mu` and `log_var`.
4. Clamps `log_var ∈ [-6, 4]` to prevent KL explosion.
5. Reparameterizes: `z = mu + eps * exp(0.5 * log_var)`.

**`split_z(z: (B,d_z)) → (z_presence, z_type, z_proximity, z_noise)`**
Returns semantic blocks with activations applied (sigmoid for presence/proximity, softmax for type).

**`split_z_raw(z: (B,d_z)) → (z_presence, z_type, z_proximity, z_noise)`**
Returns raw logits without sigmoid/softmax. Used as input to downstream task heads.

---

### `scm.py`

#### `SCM` (nn.Module)
Differentiable structural causal model. Learns a soft adjacency matrix `A` over the `d_z` latent variables and a per-variable mechanism MLP `f_i`.

**`adjacency() → Tensor (d_z, d_z)`**
Returns `sigmoid(A_raw)` with diagonal zeroed (no self-loops).

**`acyclicity_loss() → scalar`**
NOTEARS acyclicity constraint: `h(A) = tr(exp(A∘A)) - d_z`. Equals zero for a perfect DAG; positive otherwise. Computed on CPU when running on MPS.

**`forward(z: (B,d_z), intervention_mask=None) → (B, d_z)`**
For each variable `i`:
1. Computes weighted parent input: `sum_j A[j,i] * z_j`.
2. Passes through `f_i(z, parent_input)` → predicted value `z_scm_i`.
3. If `intervention_mask[:, i] = 1`, replaces with the encoder's `z_i` (intervention overrides the causal mechanism).

---

### `intervention.py`

#### `KnownInterventionHandler` (nn.Module)

**`make_mask(interv_idx: (B,), device) → (B, d_z)`**
Converts integer intervention indices (0 = none, 1–7 = noise type) to a binary mask over latent dimensions. Noise intervention `k` sets `noise_dim[(k-1) % d_z_noise]` to 1. The presence, type, and proximity dimensions are never masked here.

#### `UnknownInterventionClassifier` (nn.Module)
CITRIS-style classifier for identifying which latent variable changed between two consecutive time steps.

**`forward(z_t: (B,d_z), z_t1: (B,d_z)) → (B, d_z+1)`**
Concatenates `[z_t, z_t1 - z_t]` → `(B, 2*d_z)`. MLP outputs logits over `d_z` variables + one "no intervention" class.

---

### `decoder.py`

#### `SpectralDecoder` (nn.Module)
Reconstructs filterbank log-envelopes from latent `z`. Reconstruction in filterbank space avoids phase ambiguity inherent in raw waveform reconstruction.

**`forward(z: (B,d_z)) → (B, K, T')`**
1. `expand`: Linear + ReLU → `(B, d_model)`.
2. `decode`: Linear → GELU → Linear → `(B, K*T')`.
3. Reshape to `(B, K, T')`.

---

### `downstream.py`

#### `PresenceHead` (nn.Module)
Single linear layer: `d_z_presence → 1`. Outputs a raw logit for `BCEWithLogitsLoss`.

#### `TypeHead` (nn.Module)
Shallow MLP: `d_z_type → 32 (ReLU, Dropout 0.3) → n_classes`. For `CrossEntropyLoss`.

#### `VehicleDetectionHead` (nn.Module)

**`forward(z_presence: (B,1), z_type: (B,4)) → (presence_logit: (B,1), type_logits: (B,4))`**
Passes raw (un-activated) semantic blocks to their respective heads.

---

## 5. Loss Functions (`crl_vehicle/losses/`)

### `elbo.py`

#### `reconstruction_loss(x_hat: (B,K,T'), x_target: (B,K,T')) → scalar`
MSE between decoder output and filterbank target. `reduction='mean'` over all elements.

#### `kl_divergence(mu, log_var, intervention_mask=None) → scalar`
Per-element KL: `-0.5 * (1 + log_var - mu² - exp(log_var))`. Intervened dimensions are zeroed before averaging, so interventions are not penalized for departing from the prior.

### `causal.py`

#### `scm_consistency_loss(z: (B,d_z), z_hat_scm: (B,d_z)) → scalar`
MSE between encoder-sampled `z` and the SCM's causal prediction `z_scm`. Drives the SCM to explain the encoder's output. = 0 when SCM perfectly reconstructs latent structure.

### `disentangle.py`

#### `total_correlation_loss(z: (B,d_z)) → scalar`
Computes the mean squared value of off-diagonal entries of the normalized `z` correlation matrix. = 0 when all dimensions are uncorrelated. A differentiable proxy for the Total Correlation term in beta-TCVAE.

### `combined.py`

#### `CombinedLoss` (nn.Module)

**`update_beta(epoch: int)`**
Linearly anneals KL weight from `beta_start` (0.0) to `beta_end` (4.0) over `beta_anneal_epochs`. Prevents posterior collapse in early training by deprioritizing KL before the decoder has learned useful structure.

**`_modality_terms(outputs, mod, interv_mask, beta) → (loss_tensor, metrics_dict)`**
Per-modality ELBO + causal consistency + disentanglement penalty. Returns zeros for unavailable modalities.

**`forward(outputs, beta_override=None) → (total_loss, metrics_dict)`**
Aggregates all terms:
```
L = L_audio + L_seismic
  + lambda_acyclic  * acyclicity_loss
  + lambda_interv   * intervention_identification_loss
  + lambda_task     * (vehicle_classification_loss + detection_loss)
```
`beta_override` allows validation to use a fixed beta for checkpoint comparison (epoch-invariant metric).

---

## 6. Training Pipeline (`training/`)

### `scheduler.py`

#### `build_scheduler(optimizer, config) → LambdaLR`
Two-phase LR schedule:
1. **Warmup** (epochs 0 → `warmup_epochs`): Linear ramp from 0 → `config.lr`.
2. **Cosine annealing** (after warmup): Cosine decay with period `cosine_period`, min LR = `lr_min`.

### `trainer.py`

#### `CRLModel` (nn.Module)
Full model. Holds per-modality `filterbanks`, `ssms`, `encoders`, `decoders`, `det_heads` (all `ModuleDict`), and shared `scm`, `known_interv`, `unknown_interv`.

**`encode_modality(sensor, x: (B,C,W)) → (z, mu, log_var)` each `(B,d_z)`**
Single-modality forward through Filterbank → SSM → CausalEncoder.

**`forward_known(batch, device) → dict`**
Known-intervention forward pass. Computes intervention mask from `interv_idx`, encodes both modalities, runs SCM, decodes. Returns full outputs dict expected by `CombinedLoss`.

**`forward_unknown(batch, device) → dict`**
Unknown-intervention forward pass on consecutive pairs. Encodes `(x_t, x_t1)`, runs `UnknownInterventionClassifier`, uses stored intervention indices as pseudo-labels.

**`crl_parameters()` / `head_parameters()`**
Parameter group selectors used by the optimizer. `crl_parameters` excludes downstream heads during pre-training; `head_parameters` isolates heads for fine-tuning.

---

#### `Trainer`

**`train_crl(loader_known, loader_pairs, val_loader, epochs)`**
Main CRL pre-training loop. Per epoch:
1. Calls `_train_epoch` (mixed known + unknown batches).
2. Evaluates on val with fixed-beta checkpoint metric.
3. Checkpoints best model; applies early stopping (patience=10).
4. Logs all metrics to `crl_metrics.csv`.

**`_train_epoch(loader_known, loader_pairs, epoch) → metrics`**
Single training epoch. Curriculum: `unknown_weight` ramps from 0 → 0.5 over epochs 10–20, gradually mixing in unknown-intervention batches. Applies gradient clipping (`max_norm=1.0`).

**`_eval_crl(loader, beta_override=None) → metrics`**
Evaluation pass over a loader. Passes `beta_override=config.beta_end` to `CombinedLoss` for an epoch-invariant checkpoint metric (per scientific integrity directive).

**`train_downstream(train_loader, val_loader, epochs)`**
Freezes CRL backbone. Trains only detection/classification heads with class-weighted BCE (weight `[1.0, 5.0]`) to handle detection imbalance.

**`_eval_downstream(loader) → metrics`**
Evaluates detection accuracy and weighted F1 on a loader.

### `eval.py`

#### `compute_mig(z_samples, factor_labels, n_bins=10) → dict`
Computes Mutual Information Gap (Kim & Mnih 2018). Discretizes each `z` dimension into `n_bins`, computes mutual information between each `z_i` and each ground-truth factor, returns per-factor MIG scores (∈ [0, 1]).

#### `linear_probe_accuracy(z_train, y_train, z_val, y_val, label_name) → dict`
Fits a `LogisticRegression` on frozen `z` embeddings, evaluates on val split. Returns accuracy and weighted F1. Low probe accuracy on a semantic block (e.g., `z_type`) would indicate the encoder is not learning that factor.

#### `run_full_eval(model, train_loader, val_loader, device, primary_sensor) → dict`
Collects `z` over full train+val sets. Computes:
- MIG for `vehicle_type`, `detection`, `interv_type`
- Linear probes on `z_type`, `z_presence`, full `z`
- Detection AUC (sigmoid of `z_presence` block)

---

## 7. Training Method Summary

The system uses a **multi-objective curriculum** organized in two phases.

### Phase 1: CRL Pre-training

The objective is structured as:

```
L_total = L_ELBO(audio) + L_ELBO(seismic)
        + lambda_acyclic   * h(A)         [DAG constraint]
        + lambda_interv    * L_interv      [intervention identification]
        + lambda_task      * L_task        [detection + classification]
```

**Beta annealing** (0 → 1 over the first half of training) prevents posterior collapse. The KL divergence term is gradually weighted in, allowing the VAE to first learn a good reconstruction before being forced to match the prior.

**Checkpointing** uses a fixed `beta = beta_end` regardless of training epoch, making the checkpoint metric stationary and comparable across epochs.

**Gradient clipping** (`max_norm=1.0`) is used for numerical stability.

### Phase 2: Downstream Fine-tuning

The CRL backbone is frozen (`requires_grad=False`). Only the simple linear heads (`LinearPresenceHead`, `LinearTypeHead`) are trained on the frozen latent representations.

This two-phase approach validates that the latent representations are independently useful for downstream tasks without task signal contaminating the causal structure.

---

## 8. Validation Plan

### 8.1 Unit-Level Checks

| Check | What it validates |
|---|---|
| Shape trace through each module | No silent dimension mismatches |
| Loss is finite after forward pass | No NaN/Inf in gradients or outputs |
| Gradient flows to all parameter groups | No dead sub-networks |

### 8.2 Representation Quality

**Mutual Information Gap (MIG)**


**Linear Probe Accuracy**

- `z_proximity` should correlate with any range/distance ground truth if available.

**Cross-block interference test**

- Train a probe for `vehicle_type` on `z_presence` alone. Accuracy should be near chance.
- Train a probe for `detection` on `z_noise` alone. Accuracy should be near chance.
- This directly tests semantic isolation between blocks.

### 8.3 Causal Structure Validity

**Intervention localization test**

- Check the accuracy of the `UnknownInterventionClassifier`. High accuracy indicates that changes in the input are being correctly isolated to specific latent dimensions.

**Counterfactual consistency**

- For a fixed window, manually set `z_type` to each of the 4 vehicle classes, pass through the decoder, and inspect the reconstructed features. Each class should produce qualitatively different feature maps.

### 8.4 Downstream Task Performance

| Metric | Baseline to beat | Notes |
|---|---|---|
| Detection AUC | Majority class (AUC = 0.5) | Use `z_presence` sigmoid |
| Vehicle classification accuracy | 25% (random, 4 classes) | Use `z_type` head |
| Weighted F1 (detection) | Class-frequency weighted F1 = 0 | Especially for the minority class |

### 8.5 Ablations (for scientific validity)

| Ablation | What it isolates |
|---|---|
| Remove intervention loss (`lambda_interv=0`) | Contribution of causal matching vs. a plain VAE |
| Single modality (audio-only or seismic-only) | Modality-specific contribution; robustness to sensor dropout |
| Use `MorletFilterbank` vs `MultiScale1DFrontend` | Impact of frontend choice on disentanglement |

### 8.6 Sanity Tests for Scientific Integrity

Per the project directives:

1. **Checkpoint metric stationarity**: Plot val loss using annealing beta and fixed beta over epochs. The annealing-beta curve should trend down-then-up (non-stationary); the fixed-beta curve should be monotonically improving for a well-trained model. If they diverge dramatically, investigate beta schedule.
2. **Log both raw and fixed-reference loss**: `crl_metrics.csv` should contain both. Verify post-hoc that the best checkpoint (by fixed-beta metric) also has the best downstream task performance.
3. **Epoch-invariant comparison**: When comparing two runs (e.g., different `lambda_causal`), ensure both are evaluated at the same `beta=beta_end` to make the comparison fair.
