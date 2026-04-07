# CRL Vehicle Detection — Implementation Plan

## Module Layout

```
crl_vehicle/
├── data/
│   ├── dataset.py          # SensorDataset, window logic, intervention labels
│   └── transforms.py       # per-window normalization, augmentation
├── models/
│   ├── filterbank.py       # learnable spectral filterbank (Conv1d-based)
│   ├── ssm.py              # temporal SSM layer (S4-lite or Mamba wrapper)
│   ├── encoder.py          # causal encoder: SP features → latent z
│   ├── scm.py              # structural causal model + acyclicity constraint
│   ├── intervention.py     # known + unknown intervention modules
│   ├── decoder.py          # signal/feature reconstruction head
│   └── downstream.py       # vehicle presence + type classification heads
├── losses/
│   ├── elbo.py             # reconstruction + KL
│   ├── causal.py           # SCM consistency loss
│   ├── disentangle.py      # total correlation or MMD
│   └── combined.py         # weighted total loss + β-annealing
├── training/
│   ├── trainer.py          # main training loop
│   ├── scheduler.py        # LR warmup + cosine decay, β schedule
│   └── eval.py             # MIG score, detection metrics, disentanglement probes
└── config.py               # dataclass-based config, one per experiment
```

---

## 1. Data Pipeline (`data/`)

### Tensor shapes (establish once, use everywhere)

```
raw signal:         (C, T)          C=sensor channels, T=total samples
windowed segment:   (C, W)          W=window_size
batch:              (B, C, W)
intervention label: (B,)            int, -1=unknown, 0=none, 1..K=variable index
vehicle label:      (B,)            int, 0=absent, 1..V=vehicle type
```

### `dataset.py` — `SensorDataset`

```python
class SensorDataset(Dataset):
    def __init__(self, signal_paths, label_paths, config):
        # config fields used here:
        #   window_size: int   (e.g. 2048 samples)
        #   hop_size: int      (e.g. 1024 — 50% overlap)
        #   sample_rate: int
        #   n_channels: int

        # 1. Load all raw signals and labels into memory (or use memmap for large datasets)
        # 2. Pre-compute all (file_idx, start_sample) pairs that yield a full window
        # 3. For each window, retrieve the intervention label from the label file:
        #      - label file = time-aligned CSV with columns [timestamp, intervention_idx, vehicle_type]
        #      - assign the majority label within the window

    def __getitem__(self, idx):
        # 1. Load window x: shape (C, W), dtype float32
        # 2. Apply per-window RMS normalization (see transforms.py)
        # 3. Return dict:
        #      {
        #        'x':            tensor (C, W),
        #        'interv_idx':   int,      # which causal var was intervened on (-1=unknown, 0=none)
        #        'vehicle_type': int,      # ground truth vehicle class
        #        'segment_id':   int,      # for pairing consecutive segments in CITRIS-style training
        #      }
```

### `transforms.py` — normalization

```python
def rms_normalize(x, eps=1e-8):
    # x: (C, W)
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()   # (C, 1)
    return x / (rms + eps)

# Do NOT use global dataset statistics for normalization.
# Vehicle passage events are non-stationary by nature;
# global normalization suppresses the transient you want to detect.

# Optional augmentation (apply only during training):
def add_noise(x, snr_db_range=(10, 30)):
    # Additive white Gaussian noise at random SNR within range
    # Keeps the model robust to ambient noise floor variation
    pass

def random_time_shift(x, max_shift_frac=0.1):
    # Circular shift along time axis by random fraction of W
    # Encourages temporal invariance in encoder
    pass
```

### DataLoader construction

```python
# Two loader types needed:

# 1. Standard loader (known interventions)
loader_known = DataLoader(
    dataset,
    batch_size=config.batch_size,
    collate_fn=default_collate,     # standard, interv_idx is ground truth
)

# 2. Consecutive-pair loader (unknown intervention training, CITRIS-style)
#    Each item is a PAIR of consecutive windows (t, t+1) for the same recording.
#    The model must infer which z_i changed between them.
class ConsecutivePairDataset(Dataset):
    # Wraps SensorDataset, returns (x_t, x_{t+1}, interv_idx_t, interv_idx_{t+1})
    pass

loader_pairs = DataLoader(ConsecutivePairDataset(dataset), batch_size=config.batch_size)
```

---

## 2. SP Feature Extraction (`models/filterbank.py`, `models/ssm.py`)

### Spectral filterbank (`filterbank.py`)

**Concept:** K learnable bandpass filters, initialized at physically meaningful center frequencies. Each filter captures a different spectral "causal source candidate."

```python
class LearnableFilterbank(nn.Module):
    def __init__(self, n_channels, n_filters, filter_len, sample_rate, freq_init='log'):
        super().__init__()
        # n_filters K: number of spectral bands (e.g. 32–64)
        # filter_len L: FIR tap length (e.g. 128 at 4kHz SR → 32ms impulse response)
        # freq_init: 'log' = log-spaced center freqs, 'vehicle' = domain-specific bands

        # Initialization strategy:
        # 1. Compute K center frequencies f_k
        if freq_init == 'log':
            # log-spaced from f_min to f_max (e.g. 10 Hz to SR/2)
            f_centers = torch.logspace(log10(f_min), log10(SR/2), K)
        elif freq_init == 'vehicle':
            # Seismic vehicle bands (example for ground vibration):
            # [5-15 Hz]   body resonance
            # [15-50 Hz]  wheel-rail / road interaction
            # [50-200 Hz] engine fundamental + lower harmonics
            # [200-800Hz] upper harmonics, tire noise
            f_centers = torch.tensor([10, 20, 35, 60, 100, 150, 250, 400, 600, ...])

        # 2. Build sinc-windowed bandpass filters at each center freq
        #    kernel shape: (K * C, 1, L) for depthwise Conv1d
        kernels = sinc_bandpass(f_centers, bandwidth=f_centers/4, L=filter_len, SR=sample_rate)
        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_filters * n_channels,
            kernel_size=filter_len,
            groups=n_channels,          # depthwise: each channel filtered independently
            padding=filter_len // 2,
            bias=False
        )
        self.conv.weight = nn.Parameter(kernels)   # init from sinc, allow gradient

        # 3. Envelope extraction per band
        #    Option A: squared magnitude (simpler, always non-negative)
        #    Option B: Hilbert analytic signal magnitude (more accurate)
        # Use Option A for simplicity unless you need precise phase info.
        self.pool = nn.AvgPool1d(kernel_size=config.envelope_pool, stride=config.envelope_stride)
        # envelope_pool: e.g. 64 samples (downsamples time by 64x → manageable T')

    def forward(self, x):
        # x: (B, C, W)
        y = self.conv(x)                    # (B, K*C, W) — filtered signal per band
        y = y.pow(2)                        # squared magnitude = instantaneous power
        y = self.pool(y)                    # (B, K*C, T')  T' = W // envelope_stride
        y = torch.log1p(y)                  # log-compress: stabilizes training, mirrors perception
        return y                            # (B, K*C, T')

def sinc_bandpass(f_centers, bandwidth, L, SR):
    # Returns (K, 1, L) tensor of sinc-windowed bandpass filter kernels
    # Standard signal processing: h(n) = 2*f_hi*sinc(2*f_hi*(n-L/2)) - 2*f_lo*sinc(2*f_lo*(n-L/2))
    # Apply Hann window to reduce sidelobes
    pass
```

### SSM temporal layer (`ssm.py`)

**Concept:** The filterbank output is a sequence of band-energy envelopes over time. An SSM models the temporal dynamics — how each band's energy evolves as a vehicle approaches, passes, recedes.

```python
# Option A: Use Mamba (recommended if available)
#   pip install mamba-ssm causal-conv1d
from mamba_ssm import Mamba

class TemporalSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # d_model: matches K*C from filterbank output
        # d_state: SSM state dimension (16–64 is typical)
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: (B, K*C, T') — channels-first from filterbank
        x = x.permute(0, 2, 1)             # → (B, T', K*C)  Mamba expects (B, L, D)
        x = self.ssm(x)                    # (B, T', d_model)
        return x                           # kept in (B, T', d_model) for encoder

# Option B: S4-lite (if Mamba unavailable, or for interpretability)
#   Implement as diagonal-plus-low-rank state space model
#   Key matrices: A (diagonal, complex), B, C
#   Discretize with ZOH or bilinear method at your sample rate
#   Reference: Gu et al. 2022 "Efficiently Modeling Long Sequences with Structured State Spaces"

# Option C: Causal Transformer (simplest, less parameter-efficient)
#   Use nn.TransformerEncoder with causal mask (no future leakage)
#   Works fine for shorter sequences (T' < 512)
```

---

## 3. Causal Encoder (`models/encoder.py`)

**Goal:** Map the temporally-contextualized SP features to a structured, low-dimensional latent space `z` where each coordinate has a defined causal role.

```python
class CausalEncoder(nn.Module):
    def __init__(self, d_model, d_z_presence, d_z_type, d_z_proximity, d_z_noise):
        super().__init__()
        # d_z_presence:  1   (binary: is vehicle present?)
        # d_z_type:      V   (V vehicle classes, one-hot-ish)
        # d_z_proximity: 1   (scalar: how close is the vehicle?)
        # d_z_noise:     N   (unstructured nuisance, e.g. N=4)
        # Total d_z = 1 + V + 1 + N

        self.d_z = d_z_presence + d_z_type + d_z_proximity + d_z_noise

        # Temporal aggregation: attend over T' steps → single vector
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),   # scalar attention weight per timestep
            nn.Softmax(dim=1)        # (B, T', 1)
        )

        # Project to 2*d_z (mean + log-variance for VAE reparameterization)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * self.d_z)
        )

        # Store index slices for each semantic block
        self.presence_idx  = slice(0, d_z_presence)
        self.type_idx      = slice(d_z_presence, d_z_presence + d_z_type)
        self.proximity_idx = slice(d_z_presence + d_z_type, d_z_presence + d_z_type + d_z_proximity)
        self.noise_idx     = slice(d_z_presence + d_z_type + d_z_proximity, self.d_z)

    def forward(self, x):
        # x: (B, T', d_model)
        attn = self.attn_pool(x)            # (B, T', 1)
        ctx = (attn * x).sum(dim=1)         # (B, d_model) — soft attended summary

        out = self.proj(ctx)                # (B, 2*d_z)
        mu, log_var = out.chunk(2, dim=-1)  # each (B, d_z)

        # Reparameterize
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std                  # (B, d_z)

        return z, mu, log_var

    def split_z(self, z):
        # Decompose z into semantic components
        z_presence  = torch.sigmoid(z[:, self.presence_idx])    # (B, 1)  → [0,1]
        z_type      = torch.softmax(z[:, self.type_idx], dim=-1) # (B, V) → simplex
        z_proximity = torch.sigmoid(z[:, self.proximity_idx])   # (B, 1)  → [0,1]
        z_noise     = z[:, self.noise_idx]                      # (B, N)  → unconstrained
        return z_presence, z_type, z_proximity, z_noise

# IMPORTANT: Keep the encoder thin.
# An over-expressive encoder will learn to memorize rather than disentangle.
# If you see the KL term collapse to ~0, the encoder is too powerful → add dropout or
# reduce proj hidden dim.
```

---

## 4. SCM + Intervention Module (`models/scm.py`, `models/intervention.py`)

### Structural Causal Model (`scm.py`)

**Goal:** Learn a DAG over `z` that encodes which causal variables causally influence which others. Enforce acyclicity as a differentiable constraint.

```python
class SCM(nn.Module):
    def __init__(self, d_z, hidden_dim=32):
        super().__init__()
        self.d_z = d_z

        # Learnable weighted adjacency matrix A (d_z × d_z)
        # A[i,j] = weight of edge z_j → z_i  (i.e., column j influences row i)
        # Initialize near zero → sparse graph at the start of training
        self.A_raw = nn.Parameter(torch.zeros(d_z, d_z) + 0.01 * torch.randn(d_z, d_z))

        # Causal mechanisms: one small MLP per variable z_i
        # f_i: R^d_z → R maps parent values to z_i
        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_z, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(d_z)
        ])

    def adjacency(self):
        # Soft mask: sigmoid of A_raw, with diagonal forced to 0 (no self-loops)
        A = torch.sigmoid(self.A_raw)
        A = A * (1 - torch.eye(self.d_z, device=A.device))
        return A                            # (d_z, d_z)

    def acyclicity_loss(self):
        # NOTEARS constraint: h(A) = tr(e^{A ∘ A}) - d = 0 iff A is a DAG
        # Use matrix exponential approximation for differentiability
        A = self.adjacency()
        M = A * A                           # elementwise square
        E = torch.matrix_exp(M)             # matrix exponential
        return (torch.trace(E) - self.d_z) # scalar, = 0 for a DAG

    def forward(self, z, intervention_mask=None):
        # z: (B, d_z) — sampled latent
        # intervention_mask: (B, d_z) binary, 1 = this variable was intervened on
        #                    intervened variables bypass the causal mechanism

        A = self.adjacency()                # (d_z, d_z)
        z_hat = torch.zeros_like(z)

        for i, mech in enumerate(self.mechanisms):
            parents_i = A[i] * z            # (B, d_z) — weighted parent values
            z_hat[:, i] = mech(parents_i).squeeze(-1)

        if intervention_mask is not None:
            # Intervened variables take their encoder-sampled value, not the SCM's
            z_hat = z_hat * (1 - intervention_mask) + z * intervention_mask

        return z_hat                        # (B, d_z)
```

### Intervention module (`intervention.py`)

```python
class KnownInterventionHandler(nn.Module):
    # For training batches where interv_idx is ground truth (≥ 0)
    def make_mask(self, interv_idx, d_z, device):
        # interv_idx: (B,) int tensor, value in {0=none, 1..K=variable index}
        mask = torch.zeros(B, d_z, device=device)
        for b in range(B):
            if interv_idx[b] > 0:
                mask[b, interv_idx[b] - 1] = 1.0
        return mask                         # (B, d_z)

class UnknownInterventionClassifier(nn.Module):
    # CITRIS-style: given (z_t, z_{t+1}), infer which variable was intervened on
    def __init__(self, d_z, n_intervention_targets):
        super().__init__()
        # n_intervention_targets = d_z + 1 (each variable, plus "no intervention")
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_z, 64),
            nn.ReLU(),
            nn.Linear(64, n_intervention_targets)
        )

    def forward(self, z_t, z_t1):
        # z_t, z_t1: (B, d_z) — latents of consecutive segments
        delta = torch.cat([z_t, z_t1 - z_t], dim=-1)  # (B, 2*d_z)
        logits = self.classifier(delta)                # (B, n_intervention_targets)
        return logits

# Training curriculum recommendation:
# Epoch 0–10:   train only with known interventions (interv_idx >= 0)
# Epoch 10–20:  introduce unknown intervention pairs at 25% of batch
# Epoch 20+:    50/50 mix of known and unknown
# Rationale: unknown intervention classifier needs a warm-started encoder to work well
```

---

## 5. Decoder (`models/decoder.py`)

**Goal:** Reconstruct the input (or a compressed target) from `z`. This provides the reconstruction loss signal that keeps `z` informationally grounded.

```python
class SpectralDecoder(nn.Module):
    def __init__(self, d_z, d_model, K_filters, T_prime):
        super().__init__()
        # Reconstruct filterbank envelope output, not raw signal
        # Easier target, avoids high-freq phase ambiguity
        # Target shape: (B, K_filters, T')

        self.expand = nn.Linear(d_z, d_model)
        self.decode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K_filters * T_prime)
        )
        self.K = K_filters
        self.T = T_prime

    def forward(self, z):
        # z: (B, d_z)
        h = torch.relu(self.expand(z))              # (B, d_model)
        out = self.decode(h)                        # (B, K*T')
        return out.view(-1, self.K, self.T)         # (B, K, T')

# Note: Reconstructing filterbank envelopes (log-power) is sufficient for the
# ELBO signal. You do NOT need to reconstruct raw waveforms — that adds complexity
# without benefiting the causal structure of z.
```

---

## 6. Loss Functions (`losses/`)

### ELBO (`elbo.py`)

```python
def reconstruction_loss(x_hat, x_target):
    # x_hat, x_target: (B, K, T') — filterbank log-envelopes
    return F.mse_loss(x_hat, x_target, reduction='mean')

def kl_divergence(mu, log_var, intervention_mask=None):
    # Standard Gaussian prior KL: KL(q(z|x) || N(0,I))
    # intervention_mask (B, d_z): skip KL for intervened variables
    #   (their value is externally set, not sampled from the prior)
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, d_z)
    if intervention_mask is not None:
        kl = kl * (1 - intervention_mask)
    return kl.mean()
```

### Causal consistency loss (`causal.py`)

```python
def scm_consistency_loss(z, z_hat_scm):
    # z: (B, d_z) — encoder output
    # z_hat_scm: (B, d_z) — SCM's predicted z from causal mechanisms
    # Non-intervened variables should match the SCM prediction
    return F.mse_loss(z_hat_scm, z, reduction='mean')
```

### Disentanglement regularizer (`disentangle.py`)

```python
def total_correlation_loss(z):
    # Approximate total correlation using minibatch-weighted ELBO (Chen et al. 2018)
    # TC(z) = KL(q(z) || prod_j q(z_j))
    # Batch estimate: compare log q(z) against log prod_j q(z_j)
    # Implementation: use the minibatch-based estimator from FactorVAE

    # Simpler alternative: off-diagonal correlation penalty
    B, d_z = z.shape
    z_norm = (z - z.mean(0)) / (z.std(0) + 1e-8)       # normalize columns
    cov = (z_norm.T @ z_norm) / B                        # (d_z, d_z) correlation matrix
    off_diag = cov - torch.diag(cov.diag())
    return off_diag.pow(2).mean()
```

### Combined loss with annealing (`combined.py`)

```python
class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config fields:
        #   beta_start: 0.0      (KL weight at epoch 0)
        #   beta_end:   4.0      (KL weight at beta_anneal_epochs)
        #   beta_anneal_epochs: 20
        #   lambda_causal:  1.0
        #   lambda_interv:  2.0   (higher weight: intervention target is direct supervision)
        #   lambda_disent:  0.5
        #   lambda_task:    1.0
        #   lambda_acyclic: 1.0   (weight on DAG constraint — increase if graph stays cyclic)
        self.cfg = config
        self.current_beta = config.beta_start

    def update_beta(self, epoch):
        t = min(epoch / self.cfg.beta_anneal_epochs, 1.0)
        self.current_beta = self.cfg.beta_start + t * (self.cfg.beta_end - self.cfg.beta_start)

    def forward(self, outputs, targets):
        # outputs dict from model forward pass:
        #   x_hat, x_target, mu, log_var, z, z_scm,
        #   interv_logits (optional), interv_mask,
        #   vehicle_logits, vehicle_labels
        #   acyclicity

        L_recon = reconstruction_loss(outputs['x_hat'], outputs['x_target'])
        L_kl    = kl_divergence(outputs['mu'], outputs['log_var'], outputs['interv_mask'])
        L_causal = scm_consistency_loss(outputs['z'], outputs['z_scm'])
        L_disent = total_correlation_loss(outputs['z'])
        L_acyclic = outputs['acyclicity']

        L_task = F.cross_entropy(outputs['vehicle_logits'], outputs['vehicle_labels'])

        L_interv = 0.0
        if outputs.get('interv_logits') is not None:
            L_interv = F.cross_entropy(outputs['interv_logits'], outputs['interv_targets'])

        total = (
            L_recon
            + self.current_beta * L_kl
            + self.cfg.lambda_causal  * L_causal
            + self.cfg.lambda_disent  * L_disent
            + self.cfg.lambda_interv  * L_interv
            + self.cfg.lambda_task    * L_task
            + self.cfg.lambda_acyclic * L_acyclic
        )

        return total, {
            'recon': L_recon.item(), 'kl': L_kl.item(), 'causal': L_causal.item(),
            'disent': L_disent.item(), 'interv': float(L_interv),
            'task': L_task.item(), 'acyclic': L_acyclic.item(), 'total': total.item()
        }
```

---

## 7. Downstream Head (`models/downstream.py`)

```python
class VehicleDetectionHead(nn.Module):
    def __init__(self, d_z_presence, d_z_type, n_vehicle_classes):
        super().__init__()

        # Presence head: operates on z_presence
        self.presence_head = nn.Linear(d_z_presence, 1)    # → logit (BCEWithLogitsLoss)

        # Type head: operates on z_type (already softmax from encoder)
        # But pass raw z_type logits here to avoid double-softmax
        self.type_head = nn.Linear(d_z_type, n_vehicle_classes)

    def forward(self, z_presence_raw, z_type_raw):
        presence_logit = self.presence_head(z_presence_raw)  # (B, 1)
        type_logits    = self.type_head(z_type_raw)          # (B, V)
        return presence_logit.squeeze(-1), type_logits

# Validation strategy:
# 1. First test with a frozen encoder + LINEAR probe on z.
#    If linear probe accuracy is low, the latent space isn't disentangled yet.
#    Do not train the full downstream head until linear probes are reasonable.
# 2. Then fine-tune the full downstream head with a low LR.
```

---

## 8. Training Loop (`training/trainer.py`)

```python
class Trainer:
    def __init__(self, model, loss_fn, config):
        self.model    = model
        self.loss_fn  = loss_fn
        self.config   = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,               # e.g. 1e-3
            weight_decay=config.wd      # e.g. 1e-4
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.cosine_period,   # e.g. 20 epochs
            eta_min=config.lr_min       # e.g. 1e-5
        )
        # Warmup: linearly ramp LR for first config.warmup_epochs epochs
        # Implement as a LambdaLR wrapper around the above

    def train_epoch(self, loader_known, loader_pairs, epoch):
        self.model.train()
        self.loss_fn.update_beta(epoch)

        # Interleave known and pair batches per curriculum schedule
        known_weight   = 1.0
        unknown_weight = min((epoch - 10) / 10.0, 1.0) if epoch > 10 else 0.0

        for batch_known, batch_pairs in zip(loader_known, loader_pairs):
            self.optimizer.zero_grad()

            # --- Known intervention forward pass ---
            outputs_known = self.model(
                batch_known['x'],
                interv_idx=batch_known['interv_idx'],
                mode='known'
            )
            loss_known, metrics_known = self.loss_fn(outputs_known, batch_known)

            # --- Unknown intervention forward pass ---
            loss_unknown = 0.0
            if unknown_weight > 0:
                outputs_unk = self.model(
                    batch_pairs['x_t'], batch_pairs['x_t1'],
                    mode='unknown'
                )
                loss_unknown, _ = self.loss_fn(outputs_unk, batch_pairs)

            loss = known_weight * loss_known + unknown_weight * loss_unknown

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()

    def evaluate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            # Collect z, labels over entire val set
            # Compute:
            #   1. Vehicle detection AUC (presence head)
            #   2. Vehicle type accuracy (type head)
            #   3. MIG score (Mutual Information Gap — measures disentanglement)
            #   4. Linear separability of each z_i vs. each ground truth factor
            pass

# Gradient clipping (max_norm=1.0) is non-negotiable here.
# The acyclicity loss gradient can spike during early training when the graph is
# still cyclic, and will destabilize training without clipping.
```

---

## 9. Evaluation (`training/eval.py`)

```python
def compute_mig(z_samples, factor_labels):
    # Mutual Information Gap (Kim & Mnih 2018)
    # For each ground truth factor v_k, compute MI with every z_i.
    # MIG_k = (MI(z_{(1)}, v_k) - MI(z_{(2)}, v_k)) / H(v_k)
    # where (1) and (2) are the top two z indices by MI with v_k.
    # Average over factors.
    # MIG ∈ [0, 1], higher = more disentangled.
    # Use sklearn.metrics.mutual_info_score on discretized z.
    pass

def linear_probe_accuracy(z_train, y_train, z_val, y_val):
    # Fit LogisticRegression on z_train, evaluate on z_val
    # Do this separately for each semantic block: z_presence, z_type, z_proximity
    # This measures how linearly decodable each ground truth factor is from z
    pass

# Key metrics to track per epoch:
# - val_recon_loss     (should decrease and plateau)
# - val_kl             (should not collapse to 0 — sign of posterior collapse)
# - val_mig            (should increase over training)
# - val_detection_auc  (presence prediction quality)
# - val_type_acc       (classification quality)
# - acyclicity_h       (should decrease toward 0 — graph converging to DAG)
```

---

## 10. Config (`config.py`)

```python
@dataclass
class CRLConfig:
    # Data
    sample_rate:         int   = 4000
    window_size:         int   = 2048       # samples per segment
    hop_size:            int   = 1024       # 50% overlap
    n_channels:          int   = 1          # sensor channels

    # Filterbank
    n_filters:           int   = 32
    filter_len:          int   = 128
    envelope_pool:       int   = 64
    envelope_stride:     int   = 64         # T' = window_size // envelope_stride = 32

    # SSM
    d_model:             int   = 64
    d_state:             int   = 16

    # Encoder
    d_z_presence:        int   = 1
    d_z_type:            int   = 4          # number of vehicle classes
    d_z_proximity:       int   = 1
    d_z_noise:           int   = 4
    # total d_z = 10

    # SCM
    scm_hidden:          int   = 32

    # Training
    batch_size:          int   = 64
    lr:                  float = 1e-3
    lr_min:              float = 1e-5
    wd:                  float = 1e-4
    warmup_epochs:       int   = 5
    cosine_period:       int   = 20
    n_epochs:            int   = 100

    # Loss weights
    beta_start:          float = 0.0
    beta_end:            float = 4.0
    beta_anneal_epochs:  int   = 20
    lambda_causal:       float = 1.0
    lambda_interv:       float = 2.0
    lambda_disent:       float = 0.5
    lambda_task:         float = 1.0
    lambda_acyclic:      float = 1.0

    # Curriculum
    unknown_interv_start_epoch: int = 10
    unknown_interv_ramp_epochs: int = 10
```

---

## Implementation Order (suggested)

1. `config.py` + `data/dataset.py` — get data flowing end to end with dummy shapes
2. `models/filterbank.py` — verify filterbank output shape and log-envelope values
3. `models/ssm.py` — verify temporal contextualization, check for gradient flow
4. `models/encoder.py` — verify z shape, check that mu/log_var don't collapse
5. `losses/elbo.py` — overfit one batch to confirm reconstruction works
6. `models/scm.py` — add SCM, verify acyclicity_loss decreases with training
7. `models/intervention.py` (known) — add intervention mask, verify KL masking
8. `losses/combined.py` — full loss, check all terms are non-zero
9. `models/downstream.py` — add detection head, verify task loss
10. `models/intervention.py` (unknown) — add pair loader + classifier, last stage
11. `training/eval.py` — MIG + linear probes for disentanglement diagnosis
