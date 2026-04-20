# CRL Architecture Rebuild — Plan & Checklist

This document tracks the ground-up rebuild of the Causal Representation Learning (CRL) pipeline, moving away from NOTEARS / learnable SCMs and towards CITRIS-style temporal intervention matching.

## Architecture Decisions
* **Latent Space:** presence, type, proximity, and noise.
* **Causality Mechanism:** Interventional Matching using $t$ and $t+1$ temporal pairs to isolate changes in specific latent blocks.
* **Modality Fusion:** 
  * *Early Fusion:* Multi-Scale 1D CNN frontend (captures transients and multiple frequencies natively).
  * *Late Fusion:* Custom Learnable Continuous Wavelet Transform (Morlet) for maximum interpretability.

---

## Implementation Checklist

### 1. Front-End Extractors (`models/frontend.py`)
* [x] `MultiScale1DFrontend` (For Early Fusion)
  * [x] `__init__(in_channels, out_channels, kernel_sizes)`: Sets up parallel Conv1D branches.
  * [x] `forward(x)`: Processes and merges the multi-scale convolutions.
* [x] `MorletFilterbank` (For Late Fusion)
  * [x] `__init__(in_channels, out_channels, kernel_size)`: Initializes the time vector and fixed scale buffers.
  * [x] `_build_wavelet_kernels()`: Dynamically builds the real (cosine) and imaginary (sine) filters.
  * [x] `forward(x)`: Applies the kernels and computes the power envelope.

### 2. Core Causal Latent Structure (`models/latent.py`)
* [x] `CausalLatentSpace` 
  * [x] `__init__()`: Defines the index slices for presence (1), type (4), proximity (1), and noise (4).
  * [x] `split(z)`: Applies specific activations (sigmoid for presence, softmax for type, softplus for proximity) to the relevant slices and returns the blocks.

### 3. VAE Backbone (`models/encoder_decoder.py`)
* [x] `TemporalEncoder`
  * [x] `__init__()`: Defines the sequence model (e.g., Transformer/SSM) and the projection to `mu` and `log_var`.
  * [x] `forward(features)`: Contextualizes the features over time and samples the latent vector `z`.
* [x] `FeatureDecoder`
  * [x] `__init__()`: Defines the layers to expand `z` back to the sequence length.
  * [x] `forward(z)`: Reconstructs the power envelope (for the ELBO reconstruction loss).

### 4. Causal Intervention Matching (`models/intervention.py`)
* [x] `UnknownInterventionClassifier`
  * [x] `__init__(d_z)`: Defines the MLP that predicts which latent block changed.
  * [x] `forward(z_t, z_tn)`: Concatenates adjacent time steps and outputs logits for the intervention targets.

### 5. Downstream Linear Heads (`models/heads.py`)
* [x] `LinearPresenceHead`
  * [x] `forward(z_pres)`: Outputs the binary detection logit.
* [x] `LinearTypeHead`
  * [x] `forward(z_type)`: Outputs the multi-class vehicle category logits.

### 6. Losses (`losses/crl_loss.py`)
* [x] `reconstruction_loss(x_hat, x)`
* [x] `kl_divergence(mu, log_var)`
* [x] `intervention_matching_loss(interv_logits, interv_targets)`

### 7. Data Pipeline Updates (`data/dataset.py`)
* [x] `ConsecutivePairDataset`
  * [x] Update `_build_index()` to strictly group by `(dataset, vehicle, rs_node)` with `seg_key = (scene_id, run_id)` to prevent temporal leakage between disjointed runs.