# Design Spec: CRL-Train Pipeline HTML Documentation

**Date:** 2026-04-12  
**Status:** Approved

---

## Context

The `crl-train/` package implements a Causal Representation Learning (CRL) pipeline for vehicle detection from multimodal sensor data (audio + seismic). The pipeline combines learned spectral filtering, causal temporal modeling, a Structural Causal Model over a disentangled latent space, and downstream vehicle detection/classification heads.

The HTML document serves two audiences:
- **Technical teammates / researchers** (deep ML/PyTorch knowledge; want architecture specifics, loss terms, config parameters)
- **Advisors / professors** (need the "why" and "what" narrative; less concerned with implementation minutiae)

---

## Approach

Single long-scroll `index.html` with a **sticky left sidebar** containing section anchors. Advisors read top-to-bottom following the narrative arc; engineers jump directly to the section they need. All diagrams are **hand-crafted HTML/CSS** (no Mermaid, no external libraries). Self-contained — no CDN dependencies.

**Output file:** `crl-train/docs/index.html`

---

## Page Structure

### 1. Hero
- Title: "CRL-Train: Causal Representation Learning for Vehicle Detection"
- One-sentence summary of the system's purpose
- "What problem does this solve" paragraph (2–3 sentences): multimodal sensor noise, intervention robustness, structured latent space for interpretability

### 2. System Overview
- Full end-to-end pipeline diagram (hand-crafted HTML/CSS boxes + arrows)
- Flow: Parquet files → SensorDataset → [Filterbank → SSM → CausalEncoder] × 2 modalities → SCM → Decoder → Losses → Checkpoint
- One-paragraph narrative explaining the top-level design philosophy (independent per-modality encoding, shared causal graph)

### 3. Data Pipeline
- How parquet files are indexed (3-pass: group → load → flatten)
- Window extraction, RMS normalization, resampling
- Intervention injection (IDs 1–7, what each is)
- MultiHorizonPairDataset: what pairs are and why they're needed
- Diagram: file → index → window → augmentation → batch

### 4. Model Architecture
- Sub-section per component with a mini-diagram for each:
  - **LearnableFilterbank**: sinc FIR bandpass, log-compressed envelope, output shape
  - **TemporalSSM**: projection → positional embedding → causal Transformer, output shape
  - **CausalEncoder**: attention pooling → reparameterization → semantic block split
  - **SCM**: lower-triangular adjacency (acyclic by construction), per-variable MLP mechanisms
  - **Decoder**: filterbank-space reconstruction
  - **Downstream Heads**: PresenceHead (linear), TypeHead (shallow MLP)
- Note: audio and seismic run identical stacks independently; SCM is shared

### 5. Latent Space Design
- The 4 semantic blocks: presence (1-dim), type (4-dim), proximity (1-dim), noise (4-dim) → d_z = 10
- Why this structure: each block maps to a distinct ground-truth factor
- β-annealing rationale: start β=0 to allow encoding, ramp to β=1 to enforce KL pressure
- Visual: block diagram of the z vector with labeled regions

### 6. Loss Functions
- ELBO: reconstruction + β·KL (intervened dims zeroed)
- SCM consistency: MSE(z, z_scm)
- Total correlation penalty (disentanglement)
- Posterior collapse penalty
- Task losses: BCE for presence, CrossEntropy for type
- Temporal/contrast losses for pairs
- SCM graph sparsity (L1)
- Table: each term, its weight λ, its role
- Fixed-β checkpointing: why epoch-invariant metric is used for model selection

### 7. Training Schedule
- Curriculum phases table:
  - Epoch 0–10: known interventions only
  - Epoch 10–20: ramp unknown-intervention pairs
  - Epoch 20+: 50/50 known + pairs
- LR schedule: warmup (linear) → cosine annealing with restarts
- Gradient clipping (max_norm=1.0), why it's needed for SCM stability
- Early stopping: patience=25 on fixed-β metric

### 8. Evaluation & Diagnostics
- MIG (Mutual Information Gap): formula, interpretation, target range
- Linear probe accuracy: per semantic block, what low accuracy signals
- Collapse metrics: active_dim_frac, spread_ratio, active_units_frac — targets
- Noise type separation score: high in z_noise = good, low in z_vehicle = good
- Outputs: confusion matrices (PNG), per-sample CSV, metrics CSV

### 9. Configuration Reference
- Collapsible `<details>` table for `ModalityConfig` (all fields, types, defaults, purpose)
- Collapsible `<details>` table for `CRLConfig` (all fields, types, defaults, purpose)
- Default audio config and default seismic config as code blocks

---

## Visual Design

- **Color palette:** dark navy background (#0f1117), light text (#e8eaf0), accent blue (#4a9eff), accent teal (#00d4aa) for section headers
- **Sticky sidebar:** fixed left, ~220px wide, section links highlight on scroll (scroll spy via IntersectionObserver)
- **Diagrams:** CSS flexbox/grid boxes with border, arrows via pseudo-elements or unicode → characters
- **Tables:** zebra-striped, monospace for type/default columns
- **Code blocks:** dark background, syntax-colored for Python snippets
- **No external dependencies:** all CSS/JS inline or in `<style>`/`<script>` tags

---

## Verification

1. Open `crl-train/docs/index.html` directly in a browser (no server needed — no external CDN calls)
2. Verify sticky nav highlights correct section on scroll
3. Verify `<details>` config tables expand/collapse
4. Verify all diagrams render without broken arrows or overflow at 1280px width
5. Verify the page makes sense read top-to-bottom (advisor path) and by jumping to section 4 cold (engineer path)
