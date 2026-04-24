# `crl_vehicle/priors/` — VAE priors

Pluggable prior p(z) for `VAETrainingMode`. Swap the prior to change how KL regularizes the latent without changing anything else.

## Files

| File | Contents |
|---|---|
| `base.py` | `Prior(nn.Module, ABC)` — interface all priors implement |
| `standard.py` | `StandardPrior` — classical N(0, I) |
| `conditional.py` | `ConditionalPrior` — iVAE label-conditioned Gaussian |

## The interface

```python
class Prior(nn.Module, ABC):
    def kl_to_posterior(
        self,
        mu: Tensor,           # (B, d_z) posterior mean
        logvar: Tensor,       # (B, d_z) posterior log-variance
        y: Tensor | None,     # (B, 2) labels [presence, type] — optional
    ) -> Tensor:              # scalar KL, mean over batch
        ...
```

Priors are `nn.Module` subclasses so learnable priors (like `ConditionalPrior`) have their params discovered by `Trainer`'s optimizer-group logic. The `TrainingMode` pulls them into a separate optimizer group via `self.mode.parameters()`.

`y` is always plumbed through by `VAETrainingMode._kl_terms`, even when the prior ignores it (StandardPrior does). This keeps the ABC simple and lets you swap priors without re-wiring call sites.

## `StandardPrior`

Classical VAE. `p(z) = N(0, I)`. Analytical KL:

```
KL[N(μ_q, σ_q²) ∥ N(0, I)] = ½ · Σ(σ_q² + μ_q² - 1 - log σ_q²)
```

No learnable parameters. Ignores `y`. This is the default (`config.prior_type="standard"`) and the reference implementation for all comparisons.

## `ConditionalPrior` — iVAE

`p(z | y) = N(μ(y), σ²(y))` where `(μ, log σ²)` come from a small MLP that consumes one-hot encoded labels. Provides **identifiability**: under sufficient label variation, the latent z is identifiable up to component-wise transformation — a theoretical guarantee `StandardPrior` does not provide.

### Label encoding

Labels are `(B, 2)` float tensors: `[presence_bit, type_int_as_float]`. The `_encode_labels` function produces:

```
[presence_bit, onehot(type_shifted)]     # shape (B, 7)
```

Type offset `+2` maps the raw label range `{-2, -1, 0, 1, 2, 3}` (`MULTI`, `BACKGROUND`, pedestrian, light, medium, heavy) into a non-negative index range `{0, ..., 5}` so `F.one_hot` works directly.

### Architecture

```
Linear(7 → 32) → GELU → Linear(32 → 2·d_z)
```

Output splits into `(mu_p, logvar_p)`. Logvar is clamped to `[-4, 4]` to prevent degenerate priors (too-small variance collapses KL; too-large gives uninformative prior).

### Small initialization

The final Linear layer initializes with `weight *= 0.01, bias = 0` so the prior starts approximately `N(0, I)`. Without this, random init would mean the prior is outputting random `(μ, logvar)` from epoch 0 — giving the encoder a moving target before training even starts.

### Analytical KL

Full diagonal-Gaussian KL (not the truncated "to N(0,I)" form):

```
KL[N(μ_q, σ_q²) ∥ N(μ_p, σ_p²)] =
    ½ · Σ(log σ_p² − log σ_q² + (σ_q² + (μ_q − μ_p)²) / σ_p² − 1)
```

Raises if `y is None` — conditional priors require labels. `StandardPrior` accepts `y=None` and ignores it.

## Why this abstraction exists

Before Checkpoint 2, KL computation was hardcoded to `N(0, I)` in the loss function. Adding iVAE would have required conditional branches through every training step. The `Prior` ABC moves that decision out of the training loop: the factory chooses a prior based on `config.prior_type`, `VAETrainingMode` calls `self.prior.kl_to_posterior(...)` uniformly, and every other prior (including future ones — normalizing flows, Gaussian mixture, etc.) drops in with zero changes to VAE mode.
