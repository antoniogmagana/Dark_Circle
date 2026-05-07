# `crl-train` Math Reference

A topic-grouped cheatsheet of every equation implemented in `crl-train/`.
Equations are written as they appear in code, including clamps, ε-floors, and other implementation details.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $x$ | input window (per-channel time series) |
| $\hat{x}$ | reconstruction |
| $z$ | latent code |
| $\mu, \log\sigma^2$ | posterior mean / log-variance from the encoder |
| $\mu_p(y), \log\sigma_p^2(y)$ | label-conditioned prior mean / log-variance (iVAE) |
| $y$ | label / auxiliary variable (presence, type, …) |
| $B$ | batch size |
| $P$ | partners per anchor (contrastive) |
| $\tau$ | NT-Xent temperature |
| $\beta$ | KL weight (annealed) |
| $\gamma$ | focal-loss exponent |
| $w_c$ | class weight for class $c$ |
| $f$ | frequency (Hz) |
| $s$ | wavelet scale |
| $w_0$ | Morlet center frequency parameter |
| $f_s$ | sample rate |

---

## 1. Variational / ELBO Objective

### 1.1 Reparameterization

$$
z = \begin{cases} \mu + \exp\!\left(\tfrac{1}{2}\log\sigma^2\right) \cdot \epsilon, & \epsilon \sim \mathcal{N}(0, I), \quad \text{(training)} \\ \mu, & \text{(eval)} \end{cases}
$$

Sample from the posterior at train time; use the deterministic mean at eval.

### 1.2 Posterior parameterization (with clamps)

$$
\mu = \operatorname{clamp}(\mu_{\text{head}}(h),\, -10,\, 10), \qquad
\log\sigma^2 = \operatorname{clamp}(\ell_{\text{head}}(h),\, -4,\, 4)
$$

Encoder heads produce $\mu$ and $\log\sigma^2$, then clamp to prevent numerical blow-up.

### 1.3 Standard Gaussian prior KL — $\mathrm{KL}\!\left(q(z\mid x)\,\|\,\mathcal{N}(0, I)\right)$

$$
\mathrm{KL} = \frac{1}{2} \sum_{d=1}^{D_z}\!\left( \exp(\log\sigma^2_d) + \mu_d^2 - 1 - \log\sigma^2_d \right)
$$

Reported as the mean over the batch.

### 1.4 Conditional (iVAE) prior KL — $\mathrm{KL}\!\left(q(z\mid x)\,\|\,p(z\mid y)\right)$

With $p(z\mid y) = \mathcal{N}(\mu_p(y), \exp(\log\sigma_p^2(y)))$:

$$
\mathrm{KL} = \frac{1}{2} \sum_{d=1}^{D_z}\!\left( \log\sigma_{p,d}^2 - \log\sigma_d^2 + \frac{\exp(\log\sigma_d^2) + (\mu_d - \mu_{p,d})^2}{\exp(\log\sigma_{p,d}^2)} - 1 \right)
$$

Closed-form Gaussian-Gaussian KL with a learned label-conditioned prior; mean over batch.

### 1.5 Reference ELBO (checkpointing metric, $\beta = 1$)

$$
\mathcal{L}_{\text{ref-ELBO}} = \mathcal{L}_{\text{recon}} + \mathrm{KL}_{\text{raw}}
$$

Used for model selection; epoch-invariant because it ignores the running $\beta$ schedule.

### 1.6 Adaptive $\beta$ schedule

Let $\Delta_{\text{recon}} = \mathcal{L}_{\text{recon}}^{(t)} - \mathcal{L}_{\text{recon}}^{(t-1)}$ and $\delta = $ `recon_min_delta`. Update each epoch:

$$
\beta^{(t+1)} = \begin{cases}
\max(0,\, \beta^{(t)} - \Delta_\beta), & \text{if } \mathrm{KL}_{\text{raw}} < \mathrm{KL}_{\text{floor}}  \quad (\text{collapse}) \\
\min(1,\, \beta^{(t)} + \Delta_\beta), & \text{if } \Delta_{\text{recon}} < -\delta \;\text{or}\; \mathrm{KL}_{\text{raw}} > \mathrm{KL}_{\text{target}} \quad (\text{rise}) \\
\beta^{(t)}, & \text{otherwise} \quad (\text{hold})
\end{cases}
$$

Three-regime control: drop $\beta$ on posterior collapse, raise it when reconstruction is improving or KL exceeds target.

---

## 2. Contrastive Objective

### 2.1 Cosine similarity with temperature (logits)

For L2-normalized anchor $a_i$ and partner pool $\{p_j\}_{j=1}^{BP}$ (flattened batch × partners):

$$
\ell_{ij} = \frac{\langle a_i,\, p_j \rangle}{\tau}
$$

Dot product on unit vectors equals cosine similarity; division by $\tau$ sharpens the distribution.

### 2.2 NT-Xent loss

Let $\mathcal{P}_i \subset \{1, \dots, BP\}$ be the indices of $a_i$'s positives, and $n_i = \max(|\mathcal{P}_i|,\, 1)$. Then:

$$
\log p_{ij} = \ell_{ij} - \operatorname{logsumexp}_k \ell_{ik}
$$

$$
\mathcal{L}_{\text{NT-Xent}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \!\left( -\frac{1}{n_i} \sum_{j \in \mathcal{P}_i} \log p_{ij} \right)
$$

where $\mathcal{A}$ is the set of anchors that have at least one positive. Anchors with no positive are dropped before averaging.

### 2.3 Projection head $g_\phi : \mathbb{R}^{D_z} \to \mathbb{R}^{D_{\text{proj}}}$

$$
g_\phi(z) = W_2 \cdot \operatorname{GELU}(W_1 z + b_1) + b_2
$$

Output is L2-normalized before being passed to NT-Xent.

---

## 3. Probe & Classification Losses

### 3.1 Binary cross-entropy with logits (presence head)

$$
\mathcal{L}_{\text{aux-pres}} = -\frac{1}{B} \sum_i \!\left[ w^{+} \cdot y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log (1 - \sigma(\hat{y}_i)) \right]
$$

`pos_weight` $w^{+}$ rescales the positive term to handle class imbalance.

### 3.2 Standard (weighted) cross-entropy (type head, fallback)

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{\sum_i w_{y_i}} \sum_i w_{y_i} \log p_{i, y_i}, \qquad p_{i, c} = \operatorname{softmax}(\hat{z}_i)_c
$$

### 3.3 Focal cross-entropy (type head, default)

With $p_t = p_{i, y_i}$ and per-class weight $w_{y_i}$:

$$
\mathcal{L}_{\text{focal}} = \frac{\sum_i (1 - p_t)^\gamma \cdot w_{y_i} \cdot \bigl(-\log p_t\bigr)}{\max\!\left(\sum_i w_{y_i},\; 10^{-12}\right)}
$$

If `weight` is `None`, the denominator becomes $|B|$ (plain mean). The $(1 - p_t)^\gamma$ factor down-weights confident easy examples.

### 3.4 Multi-task aggregation (VAE mode)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathrm{KL} + \lambda_{\text{interv}} \mathcal{L}_{\text{interv}} + \lambda_{\text{aux-pres}} \mathcal{L}_{\text{aux-pres}} + \lambda_{\text{aux-type}} \mathcal{L}_{\text{aux-type}}
$$

Five $\lambda$ knobs control the trade-off; $\mathrm{KL}$ here is the $\beta$-scaled term used for backprop.

### 3.5 Per-sensor averaging

When $n_{\text{active}}$ sensors contribute to a forward pass:

$$
\mathcal{L}_{\text{total}} \leftarrow \frac{1}{n_{\text{active}}} \mathcal{L}_{\text{total}}, \qquad \text{same for each component in } \{\text{recon, kl, raw\_kl, interv, total}\}
$$

### 3.6 Downstream dual-head loss (independent optimizers)

$$
\mathcal{L}_{\text{pres}} = \frac{1}{\max(N_{\text{pres}}, 1)} \sum_i \operatorname{BCE}(\hat{y}_i^{\text{pres}}, y_i^{\text{pres}}), \qquad
\mathcal{L}_{\text{type}} = \frac{1}{\max(N_{\text{type}}, 1)} \sum_i \operatorname{CE}(\hat{y}_i^{\text{type}}, y_i^{\text{type}})
$$

Backpropagated into separate optimizers (the `pres` and `type` heads do not share gradient updates).

---

## 4. Disentanglement Auxiliary Losses

### 4.1 Cross-modal alignment (audio ↔ seismic signal latents)

$$
\mathcal{L}_{\text{align}} = \frac{1}{B} \sum_i \| \mu^{\text{sig}}_{\text{audio}, i} - \mu^{\text{sig}}_{\text{seismic}, i} \|_2^2
$$

Pulls the signal-subspace estimates from the two sensor modalities together.

### 4.2 Temporal stability (slow environment latent)

For the subset of consecutive-window pairs (mask $\mathcal{V}$):

$$
\mathcal{L}_{\text{stab}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \| \mu^{\text{env}}_{t_i} - \mu^{\text{env}}_{t_i + n} \|_2^2
$$

Returns 0 if no valid pairs exist.

### 4.3 Intervention invariance (signal robust to noise)

$$
\mathcal{L}_{\text{inv}} = \frac{1}{B} \sum_i \| \mu^{\text{sig}}_{\text{clean}, i} - \mu^{\text{sig}}_{\text{intervened}, i} \|_2^2
$$

Penalizes signal-latent drift under noise interventions.

### 4.4 Intervention matching (BCE over change bits)

$$
\mathcal{L}_{\text{interv}} = \operatorname{BCE\text{-}with\text{-}logits}\bigl( \hat{c}_i \in \mathbb{R}^2,\; c_i^{\text{true}} \in \{0, 1\}^2 \bigr)
$$

Two output bits per sample: `[pres_changed, type_changed]`.

---

## 5. Frontend Math

### 5.1 Morlet wavelet kernel

Time grid (samples → seconds):

$$
t = \frac{\operatorname{linspace}\!\left(-\lfloor k_s / 2 \rfloor,\, \lfloor k_s / 2 \rfloor,\, k_s\right)}{f_s}
$$

Real and imaginary kernel components for scale $s$ and center frequency $w_0$:

$$
\psi_{\text{re}}(t; s, w_0) = (\pi s)^{-1/4} \exp\!\left(-\frac{t^2}{2 s^2}\right) \cos\!\left(\frac{w_0 t}{s}\right)
$$

$$
\psi_{\text{im}}(t; s, w_0) = (\pi s)^{-1/4} \exp\!\left(-\frac{t^2}{2 s^2}\right) \sin\!\left(\frac{w_0 t}{s}\right)
$$

Kernels are broadcast over the input channel dimension.

### 5.2 Scale ↔ frequency conversion

$$
s = \frac{w_0}{2 \pi f}, \qquad f = \frac{w_0}{2 \pi s}
$$

Initial scales come from a log-spaced frequency grid:

$$
\{f_k\}_{k=1}^{C_{\text{out}}} = \operatorname{logspace}\!\left( \log_{10} f_{\min},\; \log_{10} f_{\max},\; C_{\text{out}} \right)
$$

### 5.3 Learnable Morlet parameterization

$$
s_k = \exp(\ell_k), \qquad \ell_k = \log s_k \in \mathbb{R} \;\;(\text{learnable})
$$

Optionally, $w_{0,k}$ is also learned per filter; otherwise $w_{0,k} = w_0$ for all $k$.

### 5.4 Power and phase outputs

Power-only:

$$
\operatorname{out} = \log\!\left(1 + \psi_{\text{re}}(x)^2 + \psi_{\text{im}}(x)^2\right)
$$

With phase (channels concatenated):

$$
\operatorname{mag} = \sqrt{\psi_{\text{re}}(x)^2 + \psi_{\text{im}}(x)^2 + 10^{-8}}
$$

$$
\operatorname{out} = \bigl[\, \log(1 + \operatorname{mag}^2),\;\; \psi_{\text{re}}(x) / \operatorname{mag},\;\; \psi_{\text{im}}(x) / \operatorname{mag} \,\bigr]
$$

The cos/sin form avoids the wraparound discontinuity of $\operatorname{atan2}$.

### 5.5 FFT-accelerated convolution

For input length $L$ and kernel length $k_s$, choose $n_{\text{fft}}$ as the smallest power of 2 with $n_{\text{fft}} \geq L + k_s - 1$. Then:

$$
Y_{\text{re}} = \mathcal{F}^{-1}\!\left( X(f) \cdot \overline{K_{\text{re}}(f)} \right), \qquad
Y_{\text{im}} = \mathcal{F}^{-1}\!\left( X(f) \cdot \overline{K_{\text{im}}(f)} \right)
$$

Outputs are circularly shifted by $\lfloor k_s / 2 \rfloor$ and truncated to length $L$ to match `conv1d(padding = ks // 2)`.

### 5.6 Multi-scale conv frontend

For each $(k_s^{(b)}, \text{stride}^{(b)})$ branch:

$$
h^{(b)} = \operatorname{GELU}\!\left( \operatorname{GroupNorm}\!\left( \operatorname{Conv1d}_{k_s^{(b)},\,\text{stride}^{(b)}}(x) \right) \right)
$$

(optional `AdaptiveAvgPool1d(target_tokens)`), then fuse:

$$
\operatorname{out} = \operatorname{Conv1d}_{1 \times 1}\!\left( \operatorname{concat}_b h^{(b)} \right)
$$

---

## 6. Encoder / Decoder Math

### 6.1 Transformer encoder

$$
h_0 = \operatorname{LayerNorm}(W_{\text{in}} x), \qquad h_L = \operatorname{TransformerEncoder}(h_0)
$$

with `dim_feedforward = 2 · d_model`, `norm_first = True`. Final pooled embedding is the temporal mean:

$$
\bar{h} = \frac{1}{T} \sum_t h_{L, t} \in \mathbb{R}^{d_{\text{model}}}
$$

### 6.2 MLP decoder

$$
\hat{x} = \operatorname{reshape}\!\left( W_2 \cdot \operatorname{GELU}(W_1 z + b_1) + b_2,\; (B,\, C_{\text{out}},\, T) \right)
$$

### 6.3 Causal latent partition (`CausalLatentSpace`)

Indices into the latent vector $z \in \mathbb{R}^{D_z}$:

$$
z_{\text{pres}} = z_{[0:4)}, \quad z_{\text{type}} = z_{[4:10)}, \quad z_{\text{prox}} = z_{[10:13)}, \quad z_{\text{env}} = z_{[13:19)}, \quad z_{\text{free}} = z_{[19:D_z)}
$$

Total causal width = 19; any remaining dims are "free".

### 6.4 Split latent partition (`SplitLatentSpace`)

$$
z_{\text{signal}} = z_{[0:\, d_{\text{signal}})}, \qquad z_{\text{env}} = z_{[d_{\text{signal}}:\, D_z)}
$$

---

## 7. Augmentations & Signal Transforms

### 7.1 DC removal

$$
\tilde{x}_c[t] = x_c[t] - \frac{1}{T} \sum_\tau x_c[\tau]
$$

Per-channel mean subtraction along the time axis.

### 7.2 RMS-based noise scaling (intervention augmentation)

Let $\operatorname{RMS}(u) = \sqrt{\overline{u^2}}$, both clamped at $10^{-8}$.

$$
\hat{n} = n \cdot \frac{0.2 \cdot \operatorname{RMS}(x)}{\operatorname{RMS}(n)}, \qquad x_{\text{noisy}} = x + \hat{n}
$$

Noise is scaled to 20% of signal RMS; for batched inputs, RMS is computed per sample.

### 7.3 Brown noise (cumulative-sum white)

$$
n[k] = \frac{1}{\sqrt{N}} \sum_{j=1}^k w_j, \qquad w_j \sim \mathcal{N}(0, 1)
$$

### 7.4 Pink noise ($1/\sqrt{f}$ shaping)

Let $W(f) = \mathcal{F}\{w\}$ for white noise $w$ and $f$ the rfft frequency grid clamped at $10^{-6}$:

$$
N(f) = \frac{W(f)}{\sqrt{f}}, \qquad n = \mathcal{F}^{-1}\{N(f)\}
$$

### 7.5 Green noise (300–800 Hz bandpass)

$$
N(f) = W(f) \cdot \mathbb{1}_{300 \leq f \leq 800}, \qquad n = \mathcal{F}^{-1}\{N(f)\}
$$

### 7.6 Low-frequency oscillation

$$
n[t] = \sin(2\pi f t), \qquad f \sim \mathcal{U}[2, 12]\;\text{Hz}, \qquad t = \operatorname{linspace}(0,\, N / f_s,\, N)
$$

### 7.7 High-frequency linear chirp

With $f_0 = 1000\,\text{Hz}$, $f_1 = \min(f_s / 2 - 1,\, 8000)\,\text{Hz}$, and $k = (f_1 - f_0) / (N / f_s)$:

$$
n[t] = \cos\!\left( 2\pi \!\left( f_0 t + \tfrac{1}{2} k t^2 \right) \right)
$$

Instantaneous frequency $f(t) = f_0 + k t$.

### 7.8 Bird-chirp (Gaussian bandpass at 2–4 kHz)

With $f_c \sim \mathcal{U}[2000, 4000]\,\text{Hz}$ and bandwidth $\sigma_f = 500\,\text{Hz}$:

$$
M(f) = \exp\!\left( -\tfrac{1}{2} \!\left(\frac{f - f_c}{\sigma_f}\right)^{\!2} \right), \qquad n = \mathcal{F}^{-1}\{ W(f) \cdot M(f) \}
$$

---

## 8. Optimization Schedules

### 8.1 Cosine annealing (Stage-1, PyTorch built-in)

$$
\eta_t = \eta_{\min} + \tfrac{1}{2} (\eta_0 - \eta_{\min}) \!\left( 1 + \cos\!\left( \frac{\pi t}{T_{\max}} \right) \right)
$$

`CosineAnnealingLR(T_max = n_epochs, eta_min = lr_min)`.

### 8.2 Stage-2 schedule (warmup + cosine, multiplier form)

Cosine multiplier with floor $r_{\min} = $ `lr_min_ratio`:

$$
m_{\cos}(e, T) = r_{\min} + (1 - r_{\min}) \cdot \tfrac{1}{2} \!\left( 1 + \cos\!\left( \pi \cdot \frac{\min(\max(e/T, 0), 1)}{1} \right) \right)
$$

Per parameter group:

$$
m_{\text{learnable\_morlet}}(e) = \begin{cases}
\dfrac{e + 1}{\max(W, 1)}, & e < W \quad (\text{linear warmup}) \\
m_{\cos}(e - W,\; \max(T_{\text{tot}} - W, 1)), & e \geq W
\end{cases}
$$

$$
m_{\text{backbone, mode}}(e) = m_{\cos}(e,\, T_{\text{tot}})
$$

where $W$ = `warmup_epochs`, $T_{\text{tot}}$ = `total_epochs`. Backbone and mode groups skip warmup.

### 8.3 Gradient clipping

$$
g \leftarrow g \cdot \min\!\left( 1,\; \frac{1.0}{\|g\|_2 + \varepsilon} \right)
$$

Global L2-norm clip at $1.0$ across all model parameters.

---

## 9. Evaluation Metrics

### 9.1 Binary classification metrics

With $TP, FP, TN, FN$ from $\hat{y} = \mathbb{1}[\ell > 0]$:

$$
\text{Precision} = \frac{TP}{\max(TP + FP,\, 1)}, \qquad
\text{Recall} = \frac{TP}{\max(TP + FN,\, 1)}, \qquad
\text{Specificity} = \frac{TN}{\max(TN + FP,\, 1)}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\max(\text{Precision} + \text{Recall},\, 10^{-8})} \;=\; \frac{2 TP}{\max(2 TP + FP + FN,\, 1)}
$$

$$
\text{Acc} = \frac{TP + TN}{\max(N,\, 1)}, \qquad \text{BalancedAcc} = \tfrac{1}{2}(\text{Recall} + \text{Specificity})
$$

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \quad (\text{0 if denom} = 0)
$$

### 9.2 Confusion matrix

For multi-class predictions $\hat{y} = \arg\max_c \ell_c$:

$$
\operatorname{CM}[t, p] = \sum_i \mathbb{1}[y_i = t] \cdot \mathbb{1}[\hat{y}_i = p]
$$

Rows = true label, columns = predicted.

### 9.3 Per-class and macro F1

For each class $c$, compute $\text{Precision}_c, \text{Recall}_c, F_{1,c}$ as in 9.1 with one-vs-rest counts. Then:

$$
F_1^{\text{macro}} = \frac{1}{C} \sum_{c=1}^C F_{1,c}, \qquad
F_1^{\text{macro, support-only}} = \frac{1}{|\mathcal{C}_{>0}|} \sum_{c \in \mathcal{C}_{>0}} F_{1,c}
$$

where $\mathcal{C}_{>0} = \{c : \operatorname{support}_c > 0\}$. The "support-only" variant avoids penalizing splits where some classes are absent.

### 9.4 Empirical priors (binary and multi-class)

$$
\hat{p} = \operatorname{clamp}\!\left( \frac{1}{N} \sum_i y_i,\; \varepsilon,\; 1 - \varepsilon \right)
$$

$$
\hat{p}_c = \operatorname{clamp}\!\left( \frac{\#\{i : y_i = c\}}{\max\!\left(\sum_{c'} \#\{i : y_i = c'\},\, 1\right)},\; \varepsilon \right)
$$

### 9.5 Logit adjustment for prior shift

Binary log-odds shift:

$$
\hat{\ell}_i = \ell_i + \!\left( \log \frac{\hat{p}_{\text{split}}}{1 - \hat{p}_{\text{split}}} - \log \frac{\hat{p}_{\text{train}}}{1 - \hat{p}_{\text{train}}} \right)
$$

Multi-class adjustment (per-class shift):

$$
\hat{\ell}_{i, c} = \ell_{i, c} + \log \hat{p}_{\text{split}, c} - \log \hat{p}_{\text{train}, c}
$$

Used when the deployment / eval split has a different class distribution than training.

### 9.6 ε-tie-band ranking (bundle promotion)

Sort candidates by primary metric $m^{(1)}$ (descending). Form tie-bands of consecutive entries within $\varepsilon = 0.01$ of the band anchor:

$$
\text{band}_k = \!\left\{ j :\; m^{(1)}_{\text{anchor}_k} - m^{(1)}_j < \varepsilon,\; j \geq \text{anchor}_k \right\}
$$

Within each band, re-sort by tiebreaker $m^{(2)}$ (descending). Then concatenate bands in order. Prevents razor-thin primary-metric differences from dominating the ranking.

### 9.7 Promotion floors

A bundle is auto-promoted to `<kind>-default` only if:

$$
\text{detect:} \quad \text{pres}\_F_1 \geq 0.80, \qquad
\text{classify:} \quad \min_c \text{type}\_F_{1,c} \geq 0.40
$$

---

*Generated from `crl-train/` source on 2026-05-06. Equations transcribe code as written, including clamps, ε-floors, and other implementation defaults.*
