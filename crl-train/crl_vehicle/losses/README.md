# `crl_vehicle/losses/` — loss functions

Pure functions, no state. Training modes call these directly.

## Files

| File | Contents |
|---|---|
| `crl_loss.py` | `reconstruction_loss`, `kl_divergence`, `intervention_matching_loss`, `focal_cross_entropy` |
| `contrastive.py` | `nt_xent_loss` |
| `disentanglement.py` | `cross_modal_alignment_loss`, `temporal_stability_loss`, `intervention_invariance_loss` |

## `crl_loss.py` — VAE losses

### `reconstruction_loss(x_hat, x) → scalar`
MSE between reconstructed and target features. The target is the **frontend output** (not raw waveform) — the decoder reconstructs in the feature space the encoder sees. This is what makes different frontends (multiscale vs Morlet) directly comparable under the same ELBO.

### `kl_divergence(mu, log_var, beta=1.0) → scalar`
KL[q(z|x) ∥ N(0, I)], summed over latent dims and meaned over batch, scaled by β.

**Not used by `VAETrainingMode` directly** — that module goes through the `Prior` abstraction for pluggable KL (StandardPrior uses the same formula; ConditionalPrior uses a label-conditioned diagonal Gaussian). This standalone function is kept for legacy callers and as a reference implementation.

### `intervention_matching_loss(logits, targets) → scalar`
Binary cross-entropy for the 2-bit intervention-matching prediction (presence-changed, type-changed). Called from `vae_mode.py` with logits from `UnknownInterventionClassifier(z_env_t, z_env_tn)` and targets from `label_change_target(det_t, det_tn, type_t, type_tn)`.

This is the CITRIS piece: rather than learning which latent factor causes which label, we train an auxiliary classifier to identify *which latent block changed* between paired windows. The gradient flows back through the encoder, pressuring the factorization to be temporally coherent in the right blocks.

### `focal_cross_entropy(logits, target, weight=None, gamma=2.0) → scalar`
Focal cross-entropy: `(1 - p_t)^γ · weighted_CE`, mean-reduced (or weight-normalized when `weight` is given). Easy confident-correct samples (p_t near 1) contribute less; uncertain samples (p_t near 1/n_classes) contribute more. With `gamma=0` and a `weight` argument it reduces *exactly* to `F.cross_entropy(weight=weight)`, so it's a strict superset of weighted CE.

Stacked, not substituted: when `cfg.use_focal_type=True`, the downstream type loss becomes `focal_cross_entropy(weight=type_class_weights, gamma=cfg.focal_type_gamma)` — the focal modulator multiplies the inverse-frequency-weighted CE rather than replacing it. So minority classes still get extra gradient *and* hard examples within each class get extra gradient. The pretraining `aux_type` site uses `weight=None` (matching its existing unweighted behavior) and only adds the focal modulator.

## `contrastive.py` — NT-Xent

### `nt_xent_loss(anchor, partners, is_positive, temperature=0.1) → scalar`
SimCLR-style contrastive loss. Shapes:
- `anchor`: `(B, D)` L2-normalized anchor embeddings
- `partners`: `(B, P, D)` L2-normalized partner embeddings
- `is_positive`: `(B, P)` bool — which partners are positives for each anchor

Negatives are implicit: all `B·P` partners in the batch serve as candidates, with `is_positive` selecting which of the anchor's own partners count as positives. Other anchors' positives become in-batch negatives (standard SimCLR practice).

**Rows with zero positives are dropped** from the per-anchor average — avoids fake "zero loss" when `is_positive` is all-False for some anchor (which happens occasionally with stratified sampling when only `DIFF_TYPE`/`CROSS_DS` partners were available for that row).

### Temperature choice
Default 0.1 (SimCLR-standard). Lower → sharper softmax, more punishing on wrong-side similarities. Exposed via `config.contrastive_temperature`; passed into `ContrastiveTrainingMode`.

### Why `mu`, not sampled `z`?
`ContrastiveTrainingMode` calls the encoder's `mu` (posterior mean) rather than the reparameterized sample. Sampling at train time injects noise that hurts NT-Xent discriminability — the point of contrastive is a deterministic representation. The VAE path stays stochastic; they share the same encoder but call it differently.
