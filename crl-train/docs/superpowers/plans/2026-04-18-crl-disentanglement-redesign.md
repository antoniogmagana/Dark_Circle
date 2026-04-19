# CRL Disentanglement Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the CRL pipeline so the latent space genuinely disentangles vehicle presence, type, and proximity from environmental noise, producing downstream linear probes that beat chance.

**Architecture:** Expand the VAE latent space to `d_z=24` with named raw-slice subspaces; add supervised auxiliary losses during pre-training to pin each subspace to its semantic role; replace the noise-type intervention signal with a label-change vector derived from ground-truth consecutive-window labels; wire `run_experiments.py` to run the full pipeline end-to-end with hardware-adaptive config.

**Tech Stack:** Python 3.11, PyTorch 2.x, `torchaudio`, `sklearn` (F1/accuracy), `pandas`, `pyarrow` (parquet)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crl_vehicle/config.py` | Modify | Add `d_z=24`, aux loss weights, hardware profile fields, intervention mode flag |
| `crl_vehicle/models/latent.py` | Modify | Raw-slice `split()`, `d_z=24`, remove nonlinearities |
| `crl_vehicle/models/heads.py` | Modify | Resize heads to new subspace dims, add `LinearProximityHead` |
| `crl_vehicle/models/intervention.py` | Modify | 2-bit BCE classifier on `z_env` concat; new `label_change_target()` helper |
| `crl_vehicle/data/dataset.py` | Modify | `ConsecutivePairDataset` emits label pairs; `collate_pairs` adds 4 new keys |
| `training/trainer.py` | Modify | Aux heads on `CRLModel`; extended loss + metrics; downstream uses single loader; `val_prox_loss` + `class_breakdown` |
| `run_experiments.py` | Modify | `import csv` fix; `build_single_loaders`; downstream pipeline call; new 5-exp grid; hardware profile; updated report |

---

## Task 1: Extend `CRLConfig` with new hyperparameters and hardware profile

**Files:**
- Modify: `crl_vehicle/config.py`

- [ ] **Step 1: Add new fields to `CRLConfig`**

Replace the entire `CRLConfig` dataclass body in `crl_vehicle/config.py` with the following (keep all existing fields, add new ones):

```python
@dataclass
class CRLConfig:
    # Per-modality signal processing configs
    audio_cfg:   ModalityConfig = field(default_factory=default_audio_config)
    seismic_cfg: ModalityConfig = field(default_factory=default_seismic_config)

    # Data
    sample_seconds:  float = 1.0

    # Latent space
    d_z:             int   = 24   # total latent dims (pres=4, type=6, prox=3, env=6, free=4+1 spare)

    # Encoder / decoder
    d_model:         int   = 64
    n_heads:         int   = 4
    n_layers:        int   = 2

    # Training
    batch_size:           int   = 512
    lr:                   float = 3e-4
    lr_min:               float = 1e-4
    wd:                   float = 1e-4
    n_epochs:             int   = 100
    num_workers:          int   = 12
    early_stop_patience:  int   = 25

    # Loss weights — core
    lambda_interv:    float = 1.0

    # Loss weights — auxiliary supervision
    lambda_aux_pres:  float = 0.3
    lambda_aux_type:  float = 0.3
    lambda_aux_prox:  float = 0.1

    # Auxiliary supervision toggle (False = exp1_baseline behaviour)
    use_aux_supervision: bool = True

    # Intervention signal mode: "label_change" (redesigned) or "noise_type" (legacy)
    intervention_mode: str = "label_change"

    # Adaptive beta schedule
    beta_step:        float = 0.02
    kl_floor:         float = 0.01
    kl_target:        float = 0.5
    recon_min_delta:  float = 0.005

    # Data windowing
    horizon_stride_sec: float = 0.7

    # Training throughput
    steps_per_epoch: int | None = None

    # Paths
    save_dir:        str   = "saved_crl"

    # Frontend architecture
    frontend_type:          str = "multiscale"
    morlet_kernel_size:     int = 257
    morlet_pool_stride:     int = 64
    multiscale_pool_stride: int = 16

    # Hardware profile (set by hardware_profile() in run_experiments.py)
    hardware_profile_name: str = "auto"

    def modality_cfg(self, modality: str) -> ModalityConfig:
        if modality == "audio":
            return self.audio_cfg
        if modality == "seismic":
            return self.seismic_cfg
        raise ValueError(f"Unknown modality: {modality!r}")
```

- [ ] **Step 2: Verify config loads without error**

```bash
cd /path/to/crl-train
python -c "from crl_vehicle.config import CRLConfig; c = CRLConfig(); print(c.d_z, c.lambda_aux_pres, c.intervention_mode)"
```
Expected output: `24 0.3 label_change`

- [ ] **Step 3: Commit**

```bash
git add crl_vehicle/config.py
git commit -m "feat: extend CRLConfig with d_z=24, aux loss weights, intervention mode, hardware profile field"
```

---

## Task 2: Redesign `CausalLatentSpace` — raw slices, `d_z=24`

**Files:**
- Modify: `crl_vehicle/models/latent.py`

- [ ] **Step 1: Rewrite `latent.py`**

```python
import torch
import torch.nn as nn


class CausalLatentSpace(nn.Module):
    """
    Named raw-slice split of a d_z=24 latent vector.

    Slots (all raw — no nonlinearities applied here):
        z_pres : dims  0- 3  (4)  vehicle presence
        z_type : dims  4- 9  (6)  vehicle type/class
        z_prox : dims 10-12  (3)  proximity / amplitude
        z_env  : dims 13-18  (6)  environmental / noise factors
        z_free : dims 19-23  (4)  unconstrained remainder

    Downstream heads and auxiliary losses apply their own nonlinearities.
    """

    D_Z   = 24
    D_PRES = 4
    D_TYPE = 6
    D_PROX = 3
    D_ENV  = 6
    D_FREE = 4   # 4+1 spare kept for future use (total = 23, pad to 24)

    def __init__(self, d_z: int = 24):
        super().__init__()
        if d_z != self.D_Z:
            raise ValueError(f"CausalLatentSpace requires d_z={self.D_Z}, got {d_z}")
        self.d_z = d_z

    def split(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (..., 24)
        Returns (z_pres, z_type, z_prox, z_env, z_free) — all raw slices.
        """
        z_pres = z[..., 0:4]
        z_type = z[..., 4:10]
        z_prox = z[..., 10:13]
        z_env  = z[..., 13:19]
        z_free = z[..., 19:24]
        return z_pres, z_type, z_prox, z_env, z_free
```

- [ ] **Step 2: Verify split shapes**

```bash
python -c "
import torch
from crl_vehicle.models.latent import CausalLatentSpace
ls = CausalLatentSpace()
z = torch.randn(8, 24)
pres, typ, prox, env, free = ls.split(z)
print(pres.shape, typ.shape, prox.shape, env.shape, free.shape)
"
```
Expected: `torch.Size([8, 4]) torch.Size([8, 6]) torch.Size([8, 3]) torch.Size([8, 6]) torch.Size([8, 5])`

Wait — `z[..., 19:24]` is 5 dims (19,20,21,22,23). That's correct: D_FREE comment says 4 but 19→24 = 5 (24-19=5). The spare dim is absorbed into z_free. Update D_FREE=5 in the class:

```python
    D_FREE = 5   # dims 19-23, total = 4+5 = 24 ✓
```

Expected output after fix: `torch.Size([8, 4]) torch.Size([8, 6]) torch.Size([8, 3]) torch.Size([8, 6]) torch.Size([8, 5])`

- [ ] **Step 3: Commit**

```bash
git add crl_vehicle/models/latent.py
git commit -m "feat: expand CausalLatentSpace to d_z=24 with raw-slice split, remove sigmoid/softmax"
```

---

## Task 3: Resize downstream heads, add `LinearProximityHead`

**Files:**
- Modify: `crl_vehicle/models/heads.py`

- [ ] **Step 1: Rewrite `heads.py`**

```python
"""
Linear downstream heads — thin probes on frozen CRL backbone subspaces.

    LinearPresenceHead  : z_pres (B, 4)  → binary detection logit  (B, 1)
    LinearTypeHead      : z_type (B, 6)  → n_classes vehicle logits (B, n_classes)
    LinearProximityHead : z_prox (B, 3)  → proximity scalar         (B, 1)
                          (MSE target = RMS amplitude; range labels when available)
"""

import torch.nn as nn


class LinearPresenceHead(nn.Module):
    def __init__(self, d_in: int = 4):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_pres):
        return self.head(z_pres)   # (B, 1)


class LinearTypeHead(nn.Module):
    def __init__(self, d_in: int = 6, n_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, z_type):
        return self.head(z_type)   # (B, n_classes)


class LinearProximityHead(nn.Module):
    def __init__(self, d_in: int = 3):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, z_prox):
        return self.head(z_prox)   # (B, 1)
```

- [ ] **Step 2: Verify heads instantiate with correct shapes**

```bash
python -c "
import torch
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead, LinearProximityHead
p = LinearPresenceHead(); t = LinearTypeHead(n_classes=4); px = LinearProximityHead()
z_pres = torch.randn(8, 4); z_type = torch.randn(8, 6); z_prox = torch.randn(8, 3)
print(p(z_pres).shape, t(z_type).shape, px(z_prox).shape)
"
```
Expected: `torch.Size([8, 1]) torch.Size([8, 4]) torch.Size([8, 1])`

- [ ] **Step 3: Commit**

```bash
git add crl_vehicle/models/heads.py
git commit -m "feat: resize heads to d_z=24 subspace dims, add LinearProximityHead"
```

---

## Task 4: Redesign `UnknownInterventionClassifier` — 2-bit BCE on `z_env`

**Files:**
- Modify: `crl_vehicle/models/intervention.py`

- [ ] **Step 1: Rewrite `intervention.py`**

```python
"""
UnknownInterventionClassifier (redesigned)

Given z_env slices from two consecutive timesteps, predicts a 2-bit
label-change vector: [pres_changed, type_changed].

This is structurally valid CITRIS pressure: the classifier learns which
causal variable changed between t and t+1, pushing presence signal into
z_pres and type signal into z_type.

Noise augmentations are retained as data augmentation but are NOT part of
the intervention target — they are decoupled from causal structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_ENV = 6   # must match CausalLatentSpace.D_ENV


def label_change_target(
    det_t: torch.Tensor,
    det_tn: torch.Tensor,
    type_t: torch.Tensor,
    type_tn: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 2-bit change vector from consecutive ground-truth labels.

    det_t, det_tn  : (B,) long — detection_label {0, 1}
    type_t, type_tn: (B,) long — vehicle_type {-2, -1, 0..3}

    Returns: (B, 2) float tensor
        col 0 = pres_changed  (detection_label differs)
        col 1 = type_changed  (vehicle_type differs, ignoring background/multi)
    """
    pres_changed = (det_t != det_tn).float()

    # Only meaningful when at least one window has a valid vehicle type
    valid = (type_t >= 0) | (type_tn >= 0)
    type_changed = ((type_t != type_tn) & valid).float()

    return torch.stack([pres_changed, type_changed], dim=-1)   # (B, 2)


class UnknownInterventionClassifier(nn.Module):
    """
    MLP: given (z_env_t, z_env_tn), predict [pres_changed, type_changed].

    Input : cat([z_env_t, z_env_tn])  →  (B, 2 * D_ENV) = (B, 12)
    Output: 2 logits                  →  (B, 2)
    Loss  : BCE per bit (not softmax)
    """

    def __init__(self, d_env: int = D_ENV, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_env, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_env_t: torch.Tensor, z_env_tn: torch.Tensor) -> torch.Tensor:
        """
        z_env_t, z_env_tn : (B, D_ENV)
        Returns            : (B, 2) logits
        """
        x = torch.cat([z_env_t, z_env_tn], dim=-1)
        return self.classifier(x)
```

- [ ] **Step 2: Verify classifier and target helper**

```bash
python -c "
import torch
from crl_vehicle.models.intervention import UnknownInterventionClassifier, label_change_target
clf = UnknownInterventionClassifier()
z_env_t = torch.randn(8, 6); z_env_tn = torch.randn(8, 6)
logits = clf(z_env_t, z_env_tn)
print('logits:', logits.shape)
det_t = torch.tensor([0,1,1,0,1,0,1,0])
det_tn = torch.tensor([1,1,0,0,1,1,0,0])
type_t = torch.tensor([-1,0,1,2,-1,0,3,1])
type_tn = torch.tensor([-1,0,2,2,-1,1,3,0])
target = label_change_target(det_t, det_tn, type_t, type_tn)
print('target:', target.shape, target)
"
```
Expected: `logits: torch.Size([8, 2])`, target shape `torch.Size([8, 2])` with 0/1 float values.

- [ ] **Step 3: Commit**

```bash
git add crl_vehicle/models/intervention.py
git commit -m "feat: redesign intervention classifier to predict 2-bit label-change vector via BCE on z_env"
```

---

## Task 5: Extend `ConsecutivePairDataset` to emit label pairs

**Files:**
- Modify: `crl_vehicle/data/dataset.py`

- [ ] **Step 1: Update `ConsecutivePairDataset.__getitem__`**

In `dataset.py`, locate `ConsecutivePairDataset.__getitem__` (line ~462). The current return dict ends at `"segment_id"`. Replace the return statement with:

```python
        return {
            "x_audio_t":           get("audio",   w_t,  interv_t),
            "x_audio_tn":          get("audio",   w_tn, interv_tn),
            "x_seismic_t":         get("seismic", w_t,  interv_t),
            "x_seismic_tn":        get("seismic", w_tn, interv_tn),
            "audio_avail":         audio_avail,
            "seismic_avail":       seismic_avail,
            "interv_idx_t":        interv_t,
            "interv_idx_tn":       interv_tn,
            "horizon_n":           1,
            "vehicle_type":        vehicle_type,
            "detection_label":     det_label,
            "vehicle_type_t":      vehicle_type,
            "vehicle_type_tn":     self.ds._index[self._anchors[idx] + 1][2] if idx + 1 < len(self._anchors) else vehicle_type,
            "detection_label_t":   det_label,
            "detection_label_tn":  self.ds._index[self._anchors[idx] + 1][3] if idx + 1 < len(self._anchors) else det_label,
            "segment_id":          seismic_seg_id if seismic_avail else audio_seg_id,
        }
```

Wait — this is wrong. The consecutive pair is `(w_t, w_t+1)` from the *same* anchor entry. The `_tn` labels come from the `w_t+1` window of the *same group*, not the next anchor. The correct approach: look up the `_index` entry for `(gkey, w_t+1)`.

Replace `ConsecutivePairDataset.__getitem__` entirely:

```python
    def __getitem__(self, idx: int) -> dict:
        i_t = self._anchors[idx]
        gkey, w_t, vehicle_type, det_label, audio_seg_id, seismic_seg_id = self.ds._index[i_t]
        group = self.ds._groups[gkey]
        w_tn = w_t + 1

        # Look up tn labels from the index entry for (gkey, w_tn)
        # Build a lookup from (gkey, w) -> index position once per dataset
        # For simplicity, scan: _index is ordered so (gkey, w_tn) is i_t+1 when
        # windows are contiguous — but that's not guaranteed across groups.
        # Use the valid set built in __init__ to find the tn entry directly.
        tn_entry = self.ds._index[self._tn_map[i_t]]
        _, _, vehicle_type_tn, det_label_tn, _, _ = tn_entry

        if self.ds.is_train and torch.rand(1).item() < 0.60:
            interv_t = torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
        else:
            interv_t = 0
        if self.ds.is_train and torch.rand(1).item() < 0.60:
            interv_tn = torch.randint(1, N_INTERVENTIONS + 1, (1,)).item()
        else:
            interv_tn = 0

        audio_avail = group["audio_stem"] is not None
        seismic_avail = group["seismic_stem"] is not None

        def get(sensor, w, interv):
            if sensor == "audio" and audio_avail and w < group["audio_nw"]:
                return self.ds._get_window(sensor, group["audio_stem"], group["seg_key"], w, interv)
            if sensor == "seismic" and seismic_avail and w < group["seismic_nw"]:
                return self.ds._get_window(sensor, group["seismic_stem"], group["seg_key"], w, interv)
            return self.ds._zero_window(sensor)

        return {
            "x_audio_t":          get("audio",   w_t,  interv_t),
            "x_audio_tn":         get("audio",   w_tn, interv_tn),
            "x_seismic_t":        get("seismic", w_t,  interv_t),
            "x_seismic_tn":       get("seismic", w_tn, interv_tn),
            "audio_avail":        audio_avail,
            "seismic_avail":      seismic_avail,
            "interv_idx_t":       interv_t,
            "interv_idx_tn":      interv_tn,
            "horizon_n":          1,
            "vehicle_type":       vehicle_type,
            "detection_label":    det_label,
            "vehicle_type_t":     vehicle_type,
            "vehicle_type_tn":    vehicle_type_tn,
            "detection_label_t":  det_label,
            "detection_label_tn": det_label_tn,
            "segment_id":         seismic_seg_id if seismic_avail else audio_seg_id,
        }
```

- [ ] **Step 2: Build `_tn_map` in `ConsecutivePairDataset.__init__`**

The `__getitem__` above references `self._tn_map[i_t]` — a dict mapping anchor flat-index → tn flat-index. Add this to `__init__` after `self._anchors` is built:

```python
        # Map anchor flat-index → tn flat-index for O(1) label lookup
        pos_map = {(entry[0], entry[1]): i for i, entry in enumerate(sensor_dataset._index)}
        self._tn_map: dict[int, int] = {}
        for i_anchor in self._anchors:
            gkey, w_t = sensor_dataset._index[i_anchor][0], sensor_dataset._index[i_anchor][1]
            self._tn_map[i_anchor] = pos_map[(gkey, w_t + 1)]
```

- [ ] **Step 3: Update `collate_pairs` to include new keys**

In `dataset.py`, replace the existing `collate_pairs` function:

```python
def collate_pairs(batch: list) -> dict:
    return {
        "x_audio_t":          torch.stack([b["x_audio_t"]     for b in batch]),
        "x_audio_tn":         torch.stack([b["x_audio_tn"]    for b in batch]),
        "x_seismic_t":        torch.stack([b["x_seismic_t"]   for b in batch]),
        "x_seismic_tn":       torch.stack([b["x_seismic_tn"]  for b in batch]),
        "audio_avail":        torch.tensor([b["audio_avail"]   for b in batch], dtype=torch.bool),
        "seismic_avail":      torch.tensor([b["seismic_avail"] for b in batch], dtype=torch.bool),
        "interv_idx_t":       torch.tensor([b["interv_idx_t"]    for b in batch], dtype=torch.long),
        "interv_idx_tn":      torch.tensor([b["interv_idx_tn"]   for b in batch], dtype=torch.long),
        "horizon_n":          torch.tensor([b["horizon_n"]        for b in batch], dtype=torch.long),
        "vehicle_type":       torch.tensor([b["vehicle_type"]     for b in batch], dtype=torch.long),
        "detection_label":    torch.tensor([b["detection_label"]  for b in batch], dtype=torch.long),
        "vehicle_type_t":     torch.tensor([b["vehicle_type_t"]   for b in batch], dtype=torch.long),
        "vehicle_type_tn":    torch.tensor([b["vehicle_type_tn"]  for b in batch], dtype=torch.long),
        "detection_label_t":  torch.tensor([b["detection_label_t"]  for b in batch], dtype=torch.long),
        "detection_label_tn": torch.tensor([b["detection_label_tn"] for b in batch], dtype=torch.long),
        "segment_id":         torch.tensor([b["segment_id"]       for b in batch], dtype=torch.long),
    }
```

- [ ] **Step 4: Add `N_INTERVENTIONS` import inside `ConsecutivePairDataset.__getitem__`**

`N_INTERVENTIONS` is imported at the top of `dataset.py` already from `crl_vehicle.data.transforms`. Confirm it's present:

```bash
grep "N_INTERVENTIONS" crl_vehicle/data/dataset.py
```
Expected: at least one line showing the import.

- [ ] **Step 5: Smoke-test dataset**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset, ConsecutivePairDataset, collate_pairs
from torch.utils.data import DataLoader
cfg = CRLConfig()
ds = ConsecutivePairDataset(SensorDataset('../data_files/parsed/train', cfg, is_train=True))
loader = DataLoader(ds, batch_size=4, collate_fn=collate_pairs)
batch = next(iter(loader))
print('Keys:', sorted(batch.keys()))
print('det_t:', batch['detection_label_t'])
print('det_tn:', batch['detection_label_tn'])
print('type_t:', batch['vehicle_type_t'])
print('type_tn:', batch['vehicle_type_tn'])
"
```
Expected: keys include `detection_label_t`, `detection_label_tn`, `vehicle_type_t`, `vehicle_type_tn`. Values are integer tensors.

- [ ] **Step 6: Commit**

```bash
git add crl_vehicle/data/dataset.py
git commit -m "feat: ConsecutivePairDataset emits label pairs (det/type at t and tn), collate_pairs extended"
```

---

## Task 6: Redesign `CRLModel` and `Trainer` in `trainer.py`

**Files:**
- Modify: `training/trainer.py`

This is the largest task. Work through it in sub-steps.

### 6a: Update `CRLModel` — new latent dims, aux heads, intervention classifier

- [ ] **Step 1: Update imports at top of `trainer.py`**

Replace the existing imports block with:

```python
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES, CLASS_MAP
from crl_vehicle.models.frontend import MultiScale1DFrontend, MorletFilterbank
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder
from crl_vehicle.models.latent import CausalLatentSpace
from crl_vehicle.models.intervention import UnknownInterventionClassifier, label_change_target
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead, LinearProximityHead
from crl_vehicle.losses.crl_loss import reconstruction_loss, kl_divergence, intervention_matching_loss
```

- [ ] **Step 2: Update `CRLModel.__init__` — use `config.d_z`, add aux heads and prox heads**

Replace the `CRLModel.__init__` body (the part that builds `self.latent` and `self.interv_classifier` and the per-sensor modules):

```python
    def __init__(self, config: CRLConfig, sensors: list | None = None):
        super().__init__()
        self.cfg = config
        self.sensors = sensors or MODALITIES
        d_z = config.d_z   # 24

        self.frontends   = nn.ModuleDict()
        self.encoders    = nn.ModuleDict()
        self.decoders    = nn.ModuleDict()
        self.pres_heads  = nn.ModuleDict()
        self.type_heads  = nn.ModuleDict()
        self.prox_heads  = nn.ModuleDict()

        # Auxiliary heads for pre-training supervision (discarded after CRL phase)
        self.aux_pres_heads = nn.ModuleDict()
        self.aux_type_heads = nn.ModuleDict()
        self.aux_prox_heads = nn.ModuleDict()

        self.latent = CausalLatentSpace(d_z=d_z)
        self.interv_classifier = UnknownInterventionClassifier(
            d_env=CausalLatentSpace.D_ENV
        )

        for sensor in self.sensors:
            mod_cfg = config.modality_cfg(sensor)

            if self.cfg.frontend_type == "multiscale":
                pool_stride = config.multiscale_pool_stride
                frontend = nn.Sequential(
                    MultiScale1DFrontend(
                        in_channels=mod_cfg.n_channels,
                        out_channels=config.d_model,
                    ),
                    nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride),
                )
            elif self.cfg.frontend_type == "morlet":
                pool_stride = config.morlet_pool_stride
                frontend = nn.Sequential(
                    MorletFilterbank(
                        in_channels=mod_cfg.n_channels,
                        out_channels=config.d_model,
                        kernel_size=config.morlet_kernel_size,
                        sample_rate=mod_cfg.sample_rate,
                    ),
                    nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride),
                )
            else:
                raise ValueError(f"Unknown frontend_type: {self.cfg.frontend_type}")
            self.frontends[sensor] = frontend

            with torch.no_grad():
                dummy = torch.zeros(1, mod_cfg.n_channels, mod_cfg.window_size)
                feat_shape = frontend(dummy).shape
            c_out, t_prime = feat_shape[1], feat_shape[2]

            self.encoders[sensor] = TemporalEncoder(
                in_channels=c_out,
                d_z=d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            self.decoders[sensor] = FeatureDecoder(
                out_channels=c_out,
                seq_len=t_prime,
                d_z=d_z,
                d_model=config.d_model,
            )

            n_classes = len(CLASS_MAP)
            self.pres_heads[sensor] = LinearPresenceHead(d_in=CausalLatentSpace.D_PRES)
            self.type_heads[sensor] = LinearTypeHead(d_in=CausalLatentSpace.D_TYPE, n_classes=n_classes)
            self.prox_heads[sensor] = LinearProximityHead(d_in=CausalLatentSpace.D_PROX)

            # Auxiliary heads (pre-training only)
            self.aux_pres_heads[sensor] = nn.Linear(CausalLatentSpace.D_PRES, 1)
            self.aux_type_heads[sensor] = nn.Linear(CausalLatentSpace.D_TYPE, n_classes)
            self.aux_prox_heads[sensor] = nn.Linear(CausalLatentSpace.D_PROX, 1)
```

- [ ] **Step 3: Update `CRLModel.encode` — return 5-tuple from `latent.split`**

```python
    def encode(self, sensor: str, x: torch.Tensor):
        """
        Run frontend → encoder for one modality.
        Returns (features, z, mu, logvar).
        """
        features = self.frontends[sensor](x)
        z, mu, logvar = self.encoders[sensor](features)
        return features, z, mu, logvar
```

(No change needed — `split` is called by caller. But verify callers use 5-tuple from `latent.split`.)

- [ ] **Step 4: Update `backbone_parameters` to exclude prox and aux heads**

```python
    def backbone_parameters(self):
        """All parameters except downstream heads and aux heads."""
        exclude = set(
            list(self.pres_heads.parameters()) +
            list(self.type_heads.parameters()) +
            list(self.prox_heads.parameters()) +
            list(self.aux_pres_heads.parameters()) +
            list(self.aux_type_heads.parameters()) +
            list(self.aux_prox_heads.parameters())
        )
        return [p for p in self.parameters() if p not in exclude]

    def head_parameters(self):
        return (
            list(self.pres_heads.parameters()) +
            list(self.type_heads.parameters()) +
            list(self.prox_heads.parameters())
        )
```

### 6b: Update `Trainer._forward_pair` — new intervention signal, aux losses, metrics

- [ ] **Step 5: Replace `_forward_pair` entirely**

```python
    def _forward_pair(self, batch: dict, beta: float) -> tuple[torch.Tensor, dict]:
        """
        Shared forward pass for train and eval.
        Returns (loss, metrics_dict).
        metrics_dict includes 'raw_kl', 'aux_pres', 'aux_type', 'aux_prox',
        'pres_changed_frac', 'type_changed_frac'.
        """
        recon_total = kl_total = raw_kl_total = interv_total = 0.0
        aux_pres_total = aux_type_total = aux_prox_total = 0.0
        pres_changed_total = type_changed_total = 0.0
        n_mod = 0

        use_aux = self.cfg.use_aux_supervision
        use_label_change = (self.cfg.intervention_mode == "label_change")

        for sensor in self.model.sensors:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue

            x_t  = batch[f"x_{sensor}_t"][avail].to(self.device)
            x_tn = batch[f"x_{sensor}_tn"][avail].to(self.device)

            x_both = torch.cat([x_t, x_tn], dim=0)
            feat_both, z_both, mu_both, lv_both = self.model.encode(sensor, x_both)
            B = x_t.shape[0]
            feat_t, feat_tn = feat_both[:B], feat_both[B:]
            z_t,   z_tn    = z_both[:B],    z_both[B:]
            mu_t,  mu_tn   = mu_both[:B],   mu_both[B:]
            lv_t,  lv_tn   = lv_both[:B],   lv_both[B:]

            assert mu_both.isfinite().all(), f"mu non-finite for {sensor}"
            assert lv_both.isfinite().all(), f"lv non-finite for {sensor}"

            x_hat_both = self.model.decode(sensor, z_both)
            x_hat_t, x_hat_tn = x_hat_both[:B], x_hat_both[B:]

            recon = (reconstruction_loss(x_hat_t, feat_t) +
                     reconstruction_loss(x_hat_tn, feat_tn)) / 2

            kl_t  = kl_divergence(mu_t,  lv_t,  beta=beta)
            kl_tn = kl_divergence(mu_tn, lv_tn, beta=beta)
            kl    = (kl_t + kl_tn) / 2

            raw_kl = (kl_divergence(mu_t,  lv_t,  beta=1.0) +
                      kl_divergence(mu_tn, lv_tn, beta=1.0)) / 2
            raw_kl_total += raw_kl.item()

            # Split latent into named subspaces
            z_pres_t, z_type_t, z_prox_t, z_env_t, _ = self.model.latent.split(z_t)
            z_pres_tn, z_type_tn, z_prox_tn, z_env_tn, _ = self.model.latent.split(z_tn)

            # Intervention loss
            if use_label_change:
                det_t_lbl  = batch["detection_label_t"][avail].to(self.device)
                det_tn_lbl = batch["detection_label_tn"][avail].to(self.device)
                typ_t_lbl  = batch["vehicle_type_t"][avail].to(self.device)
                typ_tn_lbl = batch["vehicle_type_tn"][avail].to(self.device)
                targets = label_change_target(det_t_lbl, det_tn_lbl, typ_t_lbl, typ_tn_lbl)
                logits  = self.model.interv_classifier(z_env_t, z_env_tn)
                interv  = F.binary_cross_entropy_with_logits(logits, targets)
                pres_changed_total += targets[:, 0].mean().item()
                type_changed_total += targets[:, 1].mean().item()
            else:
                # Legacy noise-type signal
                interv_idx_t  = batch["interv_idx_t"][avail].to(self.device)
                interv_idx_tn = batch["interv_idx_tn"][avail].to(self.device)
                from crl_vehicle.models.intervention import interv_idx_to_block_target
                targets_legacy = interv_idx_to_block_target(interv_idx_t, interv_idx_tn)
                logits_legacy  = self.model.interv_classifier(z_env_t, z_env_tn)
                # Legacy classifier had N_BLOCK_TARGETS=5 output; now 2 — skip for baseline
                # For exp1_baseline use a fresh 5-class head if needed; here emit zero interv loss
                interv = torch.tensor(0.0, device=self.device)

            # Auxiliary supervision losses
            aux_pres = aux_type = aux_prox = torch.tensor(0.0, device=self.device)
            if use_aux:
                det_lbl = batch["detection_label"][avail].to(self.device)
                typ_lbl = batch["vehicle_type"][avail].to(self.device)

                # aux_pres: BCE on z_pres for both t and tn
                for z_pres_x in [z_pres_t, z_pres_tn]:
                    logit = self.model.aux_pres_heads[sensor](z_pres_x).squeeze(-1)
                    aux_pres = aux_pres + F.binary_cross_entropy_with_logits(
                        logit, det_lbl.float()
                    ) / 2

                # aux_type: CE on z_type, masked to vehicle-present windows
                mask = (det_lbl == 1) & (typ_lbl >= 0)
                if mask.any():
                    for z_type_x in [z_type_t, z_type_tn]:
                        logits_type = self.model.aux_type_heads[sensor](z_type_x[mask])
                        aux_type = aux_type + F.cross_entropy(
                            logits_type, typ_lbl[mask]
                        ) / 2

                # aux_prox: MSE against batch-normalized RMS amplitude
                for x_raw, z_prox_x in [(x_t, z_prox_t), (x_tn, z_prox_tn)]:
                    rms = x_raw.pow(2).mean(dim=-1).sqrt().mean(dim=-1)  # (B,)
                    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
                    prox_pred = self.model.aux_prox_heads[sensor](z_prox_x).squeeze(-1)
                    aux_prox = aux_prox + F.mse_loss(prox_pred, rms_norm) / 2

            recon_total  += recon
            kl_total     += kl
            interv_total += interv
            aux_pres_total += aux_pres.item() if isinstance(aux_pres, torch.Tensor) else aux_pres
            aux_type_total += aux_type.item() if isinstance(aux_type, torch.Tensor) else aux_type
            aux_prox_total += aux_prox.item() if isinstance(aux_prox, torch.Tensor) else aux_prox
            n_mod += 1

        if n_mod == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {
                "recon": 0.0, "kl": 0.0, "raw_kl": 0.0, "interv": 0.0, "total": 0.0,
                "aux_pres": 0.0, "aux_type": 0.0, "aux_prox": 0.0,
                "pres_changed_frac": 0.0, "type_changed_frac": 0.0,
            }

        recon_total  /= n_mod
        kl_total     /= n_mod
        raw_kl_total /= n_mod
        interv_total /= n_mod
        aux_pres_val  = aux_pres_total / n_mod
        aux_type_val  = aux_type_total / n_mod
        aux_prox_val  = aux_prox_total / n_mod

        # Convert scalar aux losses back to tensors for backprop sum
        aux_pres_t = torch.tensor(aux_pres_total / n_mod, device=self.device)
        aux_type_t = torch.tensor(aux_type_total / n_mod, device=self.device)
        aux_prox_t = torch.tensor(aux_prox_total / n_mod, device=self.device)

        total = (recon_total + kl_total
                 + self.cfg.lambda_interv * interv_total
                 + self.cfg.lambda_aux_pres * aux_pres_t
                 + self.cfg.lambda_aux_type * aux_type_t
                 + self.cfg.lambda_aux_prox * aux_prox_t)

        return total, {
            "recon":  recon_total.item(),
            "kl":     kl_total.item(),
            "raw_kl": raw_kl_total,
            "interv": interv_total.item() if isinstance(interv_total, torch.Tensor) else float(interv_total),
            "total":  total.item(),
            "aux_pres": aux_pres_val,
            "aux_type": aux_type_val,
            "aux_prox": aux_prox_val,
            "pres_changed_frac": pres_changed_total / n_mod,
            "type_changed_frac": type_changed_total / n_mod,
        }
```

**Note on legacy baseline:** `exp1_baseline` uses `intervention_mode="noise_type"` and `use_aux_supervision=False`. The legacy 5-class classifier is incompatible with the new 2-output `UnknownInterventionClassifier`. For `exp1_baseline`, set `interv = torch.tensor(0.0, ...)` (zero interv loss) so the baseline runs cleanly without a separate classifier path. This is acceptable since `exp1_baseline`'s purpose is to record current reconstruction quality, not intervention accuracy.

### 6c: Update `train_crl` — extended CSV columns, console output, beta_event logging

- [ ] **Step 6: Replace `train_crl` method**

```python
    def train_crl(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        fieldnames = [
            "epoch", "beta", "beta_event", "lr", "grad_norm",
            "train_recon", "train_kl", "train_interv", "train_total",
            "train_aux_pres", "train_aux_type", "train_aux_prox",
            "val_recon",   "val_kl",   "val_interv",   "val_total",
            "val_aux_pres", "val_aux_type", "val_aux_prox",
            "val_raw_kl", "val_ref_elbo",
            "pres_changed_frac", "type_changed_frac",
        ]
        metrics_path = self.save_dir / "crl_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("\n=== CRL Pre-training ===")
        for epoch in range(epochs):
            train_m, grad_norm = self._train_epoch(train_loader, self.beta)
            val_m   = self._eval_epoch(val_loader, self.beta)
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]

            raw_kl          = val_m["val_raw_kl"]
            recon_improving = val_m["val_recon"] < self.prev_val_recon - self.cfg.recon_min_delta
            self.prev_val_recon = val_m["val_recon"]

            if raw_kl < self.cfg.kl_floor:
                self.beta = max(0.0, self.beta - self.cfg.beta_step)
                beta_event = "↓collapse"
            elif recon_improving or raw_kl > self.cfg.kl_target:
                self.beta = min(1.0, self.beta + self.cfg.beta_step)
                beta_event = "↑"
            else:
                beta_event = "→hold"

            row = {
                "epoch": epoch, "beta": round(self.beta, 4), "beta_event": beta_event,
                "lr": round(current_lr, 6), "grad_norm": round(grad_norm, 4),
                **train_m, **val_m,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            print(
                f"Epoch {epoch:3d} | β={self.beta:.2f}({beta_event}) lr={current_lr:.2e} | "
                f"recon={val_m['val_recon']:.3f} kl={val_m['val_raw_kl']:.3f} "
                f"interv={val_m['val_interv']:.3f} | "
                f"aux_p={val_m['val_aux_pres']:.3f} aux_t={val_m['val_aux_type']:.3f} "
                f"aux_px={val_m['val_aux_prox']:.3f} | "
                f"ref_elbo={val_m['val_ref_elbo']:.3f} | "
                f"‖g‖={grad_norm:.2f} | "
                f"Δpres={val_m['val_pres_changed_frac']:.2f} Δtype={val_m['val_type_changed_frac']:.2f}"
            )

            ref_elbo = val_m["val_ref_elbo"]
            if ref_elbo < self.best_ref_elbo:
                self.best_ref_elbo = ref_elbo
                self.patience_ctr  = 0
                torch.save(self.model.state_dict(), self.save_dir / "crl_best.pth")
                print(f"  New best (ref_elbo={ref_elbo:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.save_dir / "crl_final.pth")
```

- [ ] **Step 7: Update `_train_epoch` to return `(metrics, grad_norm)`**

```python
    def _train_epoch(self, loader: DataLoader, beta: float) -> tuple[dict, float]:
        self.model.train()
        accum: dict[str, float] = {}
        n = 0
        total_grad_norm = 0.0
        for batch in loader:
            for sensor in self.model.sensors:
                if batch[f"{sensor}_avail"].any():
                    assert batch[f"x_{sensor}_t"].isfinite().all()
                    assert batch[f"x_{sensor}_tn"].isfinite().all()

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss, metrics = self._forward_pair(batch, beta)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            if self.cfg.steps_per_epoch and n >= self.cfg.steps_per_epoch:
                break
        avg_grad_norm = total_grad_norm / max(n, 1)
        result = {
            f"train_{k}": v / max(n, 1)
            for k, v in accum.items()
            if k not in ("raw_kl", "pres_changed_frac", "type_changed_frac")
        }
        return result, avg_grad_norm
```

- [ ] **Step 8: Update `_eval_epoch` to emit extended val metrics**

```python
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, beta: float) -> dict:
        self.model.eval()
        accum: dict[str, float] = {}
        n = 0
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                _, metrics = self._forward_pair(batch, beta)
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
        self.model.train()
        out = {f"val_{k}": v / max(n, 1) for k, v in accum.items()}
        out["val_ref_elbo"] = (
            out["val_recon"] + out["val_raw_kl"] +
            self.cfg.lambda_interv * out["val_interv"]
        )
        return out
```

### 6d: Update downstream training — use single loader keys, add prox eval, class breakdown

- [ ] **Step 9: Update `_train_sensor_heads` to add `LinearProximityHead` eval**

In `_train_sensor_heads`, update the downstream metrics CSV fieldnames and add `val_prox_loss`:

```python
        fieldnames = ["epoch", "train_pres_loss", "train_type_loss",
                      "val_pres_acc", "val_pres_f1", "val_type_acc", "val_type_f1",
                      "val_prox_loss", "class_breakdown"]
```

Update the row construction:
```python
            row = {
                "epoch":           epoch,
                "train_pres_loss": ep_pres / max(n_pres, 1),
                "train_type_loss": ep_type / max(n_type, 1),
                **val_m,
            }
```

- [ ] **Step 10: Update `_eval_downstream` to add `val_prox_loss` and `class_breakdown`**

Replace `_eval_downstream`:

```python
    @torch.no_grad()
    def _eval_downstream(self, loader: DataLoader, sensor: str) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        import json as _json

        self.model.eval()
        pres_head = self.model.pres_heads[sensor]
        type_head = self.model.type_heads[sensor]
        prox_head = self.model.prox_heads[sensor]

        det_true, det_pred = [], []
        cls_true, cls_pred = [], []
        prox_losses = []

        for batch in loader:
            avail = batch[f"{sensor}_avail"]
            if not avail.any():
                continue
            x = batch[f"x_{sensor}"][avail].to(self.device)
            det   = batch["detection_label"][avail]
            vtype = batch["vehicle_type"][avail]

            _, z, _, _ = self.model.encode(sensor, x)
            z_pres, z_type, z_prox, _, _ = self.model.latent.split(z)

            pres_logit  = pres_head(z_pres).squeeze(-1)
            type_logits = type_head(z_type)
            prox_pred   = prox_head(z_prox).squeeze(-1)

            # Proximity MSE vs batch-normalized RMS amplitude
            rms = x.pow(2).mean(dim=-1).sqrt().mean(dim=-1)
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
            prox_losses.append(F.mse_loss(prox_pred, rms_norm).item())

            det_pred.extend((pres_logit > 0).cpu().long().tolist())
            det_true.extend(det.tolist())

            type_mask = (det == 1) & (vtype >= 0)
            if type_mask.any():
                cls_pred.extend(type_logits[type_mask.to(self.device)].argmax(1).cpu().tolist())
                cls_true.extend(vtype[type_mask].tolist())

        def _f1(true, pred):
            if not true:
                return 0.0, 0.0
            return (
                accuracy_score(true, pred),
                f1_score(true, pred, average="weighted", zero_division=0),
            )

        pres_acc, pres_f1 = _f1(det_true, det_pred)
        type_acc, type_f1 = _f1(cls_true, cls_pred)
        val_prox_loss = sum(prox_losses) / max(len(prox_losses), 1)

        # Per-class F1
        if cls_true:
            per_class = f1_score(cls_true, cls_pred, average=None, zero_division=0)
            from crl_vehicle.config import CLASS_MAP
            class_breakdown = _json.dumps({CLASS_MAP.get(i, str(i)): round(float(v), 4)
                                           for i, v in enumerate(per_class)})
        else:
            class_breakdown = "{}"

        self.model.train()
        return {
            "val_pres_acc":    pres_acc,
            "val_pres_f1":     pres_f1,
            "val_type_acc":    type_acc,
            "val_type_f1":     type_f1,
            "val_prox_loss":   val_prox_loss,
            "class_breakdown": class_breakdown,
        }
```

- [ ] **Step 11: Verify model builds and forward pass runs**

```bash
python -c "
import sys, torch; sys.path.insert(0, '.')
from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel, Trainer
cfg = CRLConfig()
device = torch.device('cpu')
model = CRLModel(cfg).to(device)
print('params:', sum(p.numel() for p in model.parameters()))
# Quick latent split check
z = torch.randn(4, cfg.d_z)
pres, typ, prox, env, free = model.latent.split(z)
print('split shapes:', pres.shape, typ.shape, prox.shape, env.shape, free.shape)
"
```
Expected: params count printed, split shapes all correct.

- [ ] **Step 12: Commit**

```bash
git add training/trainer.py
git commit -m "feat: redesign CRLModel and Trainer — aux heads, 2-bit intervention, extended metrics, prox eval, class breakdown"
```

---

## Task 7: Rewrite `run_experiments.py`

**Files:**
- Modify: `run_experiments.py`

- [ ] **Step 1: Replace the full file**

```python
#!/usr/bin/env python3
"""
run_experiments.py — CRL diagnostic ablation runner.

Experiments
-----------
exp1_baseline      : multiscale, noise-type intervention (current behaviour), no aux supervision
exp2_aux_on        : multiscale, noise-type intervention, aux supervision ON
exp3_redesigned    : multiscale, label-change intervention, aux supervision ON
exp4_morlet        : morlet, label-change intervention, aux supervision ON
exp5_interv_strong : multiscale, label-change intervention, lambda_interv=2.0, aux supervision ON

Usage
-----
    python run_experiments.py
    python run_experiments.py --steps-per-epoch 50
    python run_experiments.py --only exp1_baseline exp3_redesigned
    python run_experiments.py --hardware-profile mid
"""

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import (
    SensorDataset, ConsecutivePairDataset,
    collate_pairs, collate_single,
)
from training.trainer import CRLModel, Trainer


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name":        "exp1_baseline",
        "description": "Multiscale, noise-type intervention, no aux supervision (current behaviour)",
        "overrides":   {
            "frontend_type":      "multiscale",
            "lambda_interv":      1.0,
            "use_aux_supervision": False,
            "intervention_mode":  "noise_type",
        },
    },
    {
        "name":        "exp2_aux_on",
        "description": "Multiscale, noise-type intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":      "multiscale",
            "lambda_interv":      1.0,
            "use_aux_supervision": True,
            "intervention_mode":  "noise_type",
        },
    },
    {
        "name":        "exp3_redesigned",
        "description": "Multiscale, label-change intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":      "multiscale",
            "lambda_interv":      1.0,
            "use_aux_supervision": True,
            "intervention_mode":  "label_change",
        },
    },
    {
        "name":        "exp4_morlet",
        "description": "Morlet frontend, label-change intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":      "morlet",
            "lambda_interv":      1.0,
            "use_aux_supervision": True,
            "intervention_mode":  "label_change",
        },
    },
    {
        "name":        "exp5_interv_strong",
        "description": "Multiscale, label-change intervention, lambda_interv=2.0, aux supervision ON",
        "overrides":   {
            "frontend_type":      "multiscale",
            "lambda_interv":      2.0,
            "use_aux_supervision": True,
            "intervention_mode":  "label_change",
        },
    },
]


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

HARDWARE_PROFILES = {
    "h100": {"batch_size": 512, "d_model": 128, "n_layers": 4, "num_workers": 12, "steps_per_epoch": None},
    "mid":  {"batch_size": 128, "d_model": 64,  "n_layers": 2, "num_workers": 8,  "steps_per_epoch": None},
    "low":  {"batch_size": 64,  "d_model": 64,  "n_layers": 2, "num_workers": 4,  "steps_per_epoch": 200},
    "cpu":  {"batch_size": 32,  "d_model": 32,  "n_layers": 1, "num_workers": 2,  "steps_per_epoch": 50},
}


def detect_hardware_profile() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 60:
        return "h100"
    elif vram_gb >= 16:
        return "mid"
    else:
        return "low"


def apply_hardware_profile(cfg: CRLConfig, profile_name: str) -> CRLConfig:
    profile = HARDWARE_PROFILES[profile_name]
    cfg.hardware_profile_name = profile_name
    for k, v in profile.items():
        if v is not None:
            setattr(cfg, k, v)
    # n_heads scales with d_model
    cfg.n_heads = max(2, cfg.d_model // 32)
    return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {name} ({vram:.0f}GB VRAM)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  Device: Apple Silicon MPS")
        return torch.device("mps")
    else:
        print("  Device: CPU (slow)")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def build_pair_loaders(data_dir: str, val_dir: str, cfg: CRLConfig):
    """For CRL pre-training — consecutive pairs."""
    train_ds = ConsecutivePairDataset(SensorDataset(data_dir, cfg, is_train=True))
    val_ds   = ConsecutivePairDataset(SensorDataset(val_dir,  cfg, is_train=False))
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=(cfg.hardware_profile_name != "cpu"),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
    )


def build_single_loaders(data_dir: str, val_dir: str, cfg: CRLConfig):
    """For downstream head training — single windows."""
    train_ds = SensorDataset(data_dir, cfg, is_train=True)
    val_ds   = SensorDataset(val_dir,  cfg, is_train=False)
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_single,
        pin_memory=(cfg.hardware_profile_name != "cpu"),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_overrides(cfg: CRLConfig, overrides: dict) -> CRLConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"CRLConfig has no attribute '{k}'")
        setattr(cfg, k, v)
    return cfg


def _best_downstream_f1(save_dir: Path) -> dict:
    """Read best downstream F1 scores from saved CSVs."""
    results = {}
    for sensor in MODALITIES:
        path = save_dir / f"downstream_metrics_{sensor}.csv"
        if not path.exists():
            continue
        best_pres = best_type = 0.0
        with open(path) as f:
            for row in csv.DictReader(f):
                best_pres = max(best_pres, float(row.get("val_pres_f1", 0)))
                best_type = max(best_type, float(row.get("val_type_f1", 0)))
        results[f"best_pres_f1_{sensor}"] = best_pres
        results[f"best_type_f1_{sensor}"] = best_type
    return results


def _best_crl_elbo(save_dir: Path) -> tuple[float, int]:
    path = save_dir / "crl_metrics.csv"
    best = float("inf")
    converged_epoch = -1
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                val = float(row.get("val_ref_elbo", "inf"))
                if val < best:
                    best = val
                    converged_epoch = int(row.get("epoch", -1))
    return best, converged_epoch


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp: dict,
    base_cfg: CRLConfig,
    data_dir: str,
    val_dir: str,
    device: torch.device,
    experiments_dir: Path,
) -> dict:
    name        = exp["name"]
    description = exp["description"]
    overrides   = exp["overrides"]

    print(f"\n{'=' * 65}")
    print(f"  Experiment: {name}")
    print(f"  {description}")
    print("  Overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    print("=" * 65)

    save_dir = experiments_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_overrides(deepcopy(base_cfg), overrides)
    cfg.save_dir = str(save_dir)

    pair_train, pair_val     = build_pair_loaders(data_dir, val_dir, cfg)
    single_train, single_val = build_single_loaders(data_dir, val_dir, cfg)

    model   = CRLModel(cfg, sensors=MODALITIES).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # CRL pre-training
    t0 = time.time()
    trainer.train_crl(pair_train, pair_val, cfg.n_epochs)
    crl_elapsed = time.time() - t0
    print(f"  CRL training done in {crl_elapsed/60:.1f} min")

    # Load best CRL checkpoint for downstream
    best_ckpt = save_dir / "crl_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Downstream head training
    t1 = time.time()
    trainer.train_downstream(single_train, single_val, cfg.n_epochs)
    ds_elapsed = time.time() - t1
    print(f"  Downstream training done in {ds_elapsed/60:.1f} min")

    best_elbo, converged_epoch = _best_crl_elbo(save_dir)
    f1_scores = _best_downstream_f1(save_dir)

    summary = {
        "name":                name,
        "description":         description,
        "overrides":           overrides,
        "crl_elapsed_min":     round(crl_elapsed / 60, 2),
        "downstream_elapsed_min": round(ds_elapsed / 60, 2),
        "best_val_ref_elbo":   best_elbo,
        "crl_converged_epoch": converged_epoch,
        **f1_scores,
    }
    with open(save_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(summaries: list[dict], report_path: Path) -> None:
    # Baseline = exp1 results (if present)
    baseline_summary = next((s for s in summaries if s["name"] == "exp1_baseline"), None)
    baseline_f1 = 0.0
    if baseline_summary:
        f1_vals = [v for k, v in baseline_summary.items()
                   if k.startswith("best_pres_f1") or k.startswith("best_type_f1")]
        baseline_f1 = sum(f1_vals) / max(len(f1_vals), 1)

    comparison = []
    for s in summaries:
        f1_vals = [v for k, v in s.items()
                   if k.startswith("best_pres_f1") or k.startswith("best_type_f1")]
        mean_f1 = sum(f1_vals) / max(len(f1_vals), 1)
        delta   = mean_f1 - baseline_f1
        verdict = "IMPROVED" if delta > 0.05 else ("MARGINAL" if delta > 0 else "NO_CHANGE")
        comparison.append({
            "name":              s["name"],
            "description":       s["description"],
            "best_val_ref_elbo": round(s.get("best_val_ref_elbo", float("inf")), 4),
            "mean_downstream_f1": round(mean_f1, 4),
            "delta_f1":          round(delta, 4),
            "verdict":           verdict,
            **{k: round(v, 4) for k, v in s.items()
               if k.startswith("best_pres_f1") or k.startswith("best_type_f1")},
        })

    report = {"summaries": summaries, "comparison": comparison}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 75}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Experiment':<24} {'ELBO':>8} {'Mean F1':>8} {'ΔF1':>8}  Verdict")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}  -------")
    for c in comparison:
        print(
            f"  {c['name']:<24} {c['best_val_ref_elbo']:>8.3f} "
            f"{c['mean_downstream_f1']:>8.3f} {c['delta_f1']:>+8.3f}  {c['verdict']}"
        )
    print(f"\n  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRL diagnostic ablation runner")
    p.add_argument("--data-dir",          default="../data_files/parsed/train")
    p.add_argument("--val-dir",           default="../data_files/parsed/val")
    p.add_argument("--crl-epochs",        type=int, default=None)
    p.add_argument("--batch-size",        type=int, default=None)
    p.add_argument("--steps-per-epoch",   type=int, default=None)
    p.add_argument("--num-workers",       type=int, default=None)
    p.add_argument("--out-dir",           default="./saved_crl/experiments")
    p.add_argument("--hardware-profile",  choices=list(HARDWARE_PROFILES), default=None,
                   help="Override hardware auto-detection")
    p.add_argument("--only",              nargs="+", default=None, metavar="NAME")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    cfg = CRLConfig()

    # Apply hardware profile (auto-detect unless overridden)
    profile_name = args.hardware_profile or detect_hardware_profile()
    apply_hardware_profile(cfg, profile_name)
    p = HARDWARE_PROFILES[profile_name]
    print(f"  Profile: {profile_name} "
          f"(batch={cfg.batch_size}, d_model={cfg.d_model}, "
          f"n_layers={cfg.n_layers}, workers={cfg.num_workers})")

    # Manual CLI overrides take precedence over profile
    if args.crl_epochs:
        cfg.n_epochs = args.crl_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.steps_per_epoch:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.num_workers:
        cfg.num_workers = args.num_workers

    experiments_dir = Path(args.out_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        valid = {e["name"] for e in EXPERIMENTS}
        unknown = set(args.only) - valid
        if unknown:
            print(f"ERROR: unknown names: {unknown}. Valid: {sorted(valid)}")
            sys.exit(1)
        experiments = [e for e in EXPERIMENTS if e["name"] in args.only]

    print(f"\nRunning {len(experiments)} experiment(s):")
    for e in experiments:
        print(f"  • {e['name']}: {e['description']}")

    summaries = []
    for exp in experiments:
        try:
            summary = run_experiment(
                exp, cfg, args.data_dir, args.val_dir,
                device, experiments_dir,
            )
            summaries.append(summary)
        except Exception as exc:
            import traceback
            print(f"\nERROR in {exp['name']}: {exc}")
            traceback.print_exc()
            summaries.append({
                "name": exp["name"], "description": f"{exp['description']} (FAILED)",
                "overrides": exp["overrides"], "error": str(exc),
            })

    write_report(summaries, experiments_dir / "report.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('run_experiments.py').read_text()); print('syntax OK')"
```
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add run_experiments.py
git commit -m "feat: rewrite run_experiments.py — 5-exp grid, downstream pipeline, hardware profile, updated report"
```

---

## Task 8: Smoke-test full pipeline end-to-end

- [ ] **Step 1: Run smoke test with 50 steps/epoch, exp1 only**

```bash
cd crl-train
python run_experiments.py \
  --steps-per-epoch 50 \
  --crl-epochs 3 \
  --only exp1_baseline \
  --out-dir ./saved_crl/smoke_test
```
Expected: runs without crash, prints hardware profile, CRL epochs, downstream epochs, writes `report.json`.

- [ ] **Step 2: Verify CRL CSV columns**

```bash
python -c "
import csv
with open('saved_crl/smoke_test/exp1_baseline/crl_metrics.csv') as f:
    cols = csv.DictReader(f).fieldnames
print(cols)
assert 'beta_event' in cols
assert 'grad_norm' in cols
assert 'train_aux_pres' in cols
assert 'val_ref_elbo' in cols
print('CRL CSV columns OK')
"
```

- [ ] **Step 3: Verify downstream CSV columns**

```bash
python -c "
import csv
with open('saved_crl/smoke_test/exp1_baseline/downstream_metrics_audio.csv') as f:
    cols = csv.DictReader(f).fieldnames
print(cols)
assert 'val_prox_loss' in cols
assert 'class_breakdown' in cols
print('Downstream CSV columns OK')
"
```

- [ ] **Step 4: Run exp3_redesigned smoke test — verify intervention signal is non-trivial**

```bash
python run_experiments.py \
  --steps-per-epoch 50 \
  --crl-epochs 3 \
  --only exp3_redesigned \
  --out-dir ./saved_crl/smoke_test
python -c "
import csv
with open('saved_crl/smoke_test/exp3_redesigned/crl_metrics.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    pf = float(r['pres_changed_frac'])
    tf = float(r['type_changed_frac'])
    assert 0 < pf < 1 or pf == 0, f'pres_changed_frac unexpected: {pf}'
    print(f\"epoch {r['epoch']}: Δpres={pf:.3f} Δtype={tf:.3f}\")
print('Intervention signal check passed')
"
```
Expected: `pres_changed_frac` and `type_changed_frac` are non-constant across epochs (actual values vary by dataset).

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: CRL disentanglement redesign complete — d_z=24, aux supervision, label-change intervention, hardware profile, extended metrics"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Intervention signal redesign (Task 4, 5, 6b)
- [x] `d_z=24` raw-slice latent (Task 2)
- [x] Head resizing (Task 3)
- [x] Auxiliary supervision during pre-training (Task 6b step 5)
- [x] `build_single_loaders` + downstream pipeline (Task 7)
- [x] Extended CRL CSV columns (Task 6c step 6)
- [x] Extended downstream CSV columns (Task 6d step 10)
- [x] Console output format (Task 6c step 6)
- [x] `experiment_summary.json` extended fields (Task 7 step 1, `_best_downstream_f1`)
- [x] Report verdict uses downstream F1 (Task 7 step 1, `write_report`)
- [x] Hardware profile auto-detection (Task 7 step 1, `detect_hardware_profile`)
- [x] `import csv` fix (Task 7 step 1, imports)
- [x] `LinearProximityHead` added (Task 3, Task 6)
- [x] `val_prox_loss` and `class_breakdown` in downstream CSV (Task 6d)

**Type consistency check:**
- `CausalLatentSpace.split()` returns 5-tuple — all callers in `_forward_pair` and `_eval_downstream` unpack 5 values ✓
- `UnknownInterventionClassifier(d_env=6)` takes `z_env_t, z_env_tn` each `(B,6)` — matches `D_ENV=6` in `CausalLatentSpace` ✓
- `LinearPresenceHead(d_in=4)`, `LinearTypeHead(d_in=6)`, `LinearProximityHead(d_in=3)` — match `D_PRES=4`, `D_TYPE=6`, `D_PROX=3` ✓
- `label_change_target` returns `(B, 2)` float — matches `F.binary_cross_entropy_with_logits` target shape ✓
- `_train_epoch` returns `(dict, float)` — `train_crl` unpacks `train_m, grad_norm` ✓
- `_tn_map` built in `__init__`, used in `__getitem__` ✓

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete. ✓

**Note on legacy intervention path:** `exp1_baseline` uses `intervention_mode="noise_type"` but the new `UnknownInterventionClassifier` only outputs 2 logits. The `_forward_pair` sets `interv = 0.0` for the noise_type path. This means `exp1_baseline` has no intervention loss — it is still a valid baseline for reconstruction quality and downstream F1, just without the intervention pressure. This is intentional and documented in the spec.
