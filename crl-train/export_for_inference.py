"""Export a trained CRLModel into TorchScript artifacts for the
inference-engine deployment pipeline.

Reads a saved run and emits per-mode TorchScript files plus a deployment-only
meta.json that the inference pods consume. The inference image has zero Python
dependency on crl_vehicle — only torch.jit.load is needed.

Two checkpoints are required (the heads are checkpointed independently):
  * downstream_best_pres.pth — argmax val_pres_f1 over training epochs
  * downstream_best_type.pth — argmax val_type_f1 over training epochs

Presence-side parameters (encoder + pres_heads + aux_pres_heads) are taken from
the pres ckpt; type-side parameters (type_heads + aux_type_heads) are taken
from the type ckpt. The encoder is shared across both heads at inference time,
so the source for it is configurable via --encoder-from (default: pres).

Bundle kinds (--bundle-kind):
  * detect   -> encoder_*.ts + meta.json (no type head). Catalog: detect-bundles/.
              meta.json adds: pres_f1, min_pres_f1, source_run.
  * classify -> encoder_*.ts + type_head_*.ts + meta.json. Catalog: classify-bundles/.
              meta.json adds: class_names, probe_mode, type_f1, min_type_f1, source_run.

A single saved CRL run produces one detect bundle and one classify bundle via
two separate invocations.

Output layout (in --out-dir):

  Per-sensor mode (frontend_type ∈ {morlet, morlet_per_sensor, morlet_learnable}):
    encoder_audio.ts        # (x_audio[B,1,16000])    -> (z[B,d_z], pres_logit[B,1])
    encoder_seismic.ts      # (x_seismic[B,1,100])    -> (z[B,d_z], pres_logit[B,1])
    type_head_audio.ts      # classify only — (z[B,d_z]) -> type_logits[B,4]
    type_head_seismic.ts    # classify only
    meta.json

  Fused mode (frontend_type ∈ {multiscale, morlet_fused, morlet_learnable_fused}):
    encoder_fused.ts        # (x_audio[B,1,16000], x_seismic[B,1,100]) -> (z, pres_logit)
    type_head_fused.ts      # classify only
    meta.json

CLI:
    # Produce a detect bundle, evaluate against the catalog, promote if winner.
    python export_for_inference.py \\
        --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \\
        --bundle-kind detect \\
        --bundle-name <frontend>-<mode>-<run>-v1 \\
        --promote-default

    # Produce a classify bundle from the same (or a different) run.
    python export_for_inference.py \\
        --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \\
        --bundle-kind classify \\
        --bundle-name <frontend>-<mode>-<run>-<probe>-v1 \\
        --promote-default

    # Escape hatch: explicit out-dir for one-off exports outside the catalog.
    python export_for_inference.py --save-dir saved_crl/runs/<run> \\
        --bundle-kind detect --out-dir /tmp/scratch-bundle
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from crl_vehicle import analysis
from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel

PER_SENSOR_FRONTENDS = {"morlet", "morlet_per_sensor", "morlet_learnable"}
FUSED_FRONTENDS = {"multiscale", "morlet_fused", "morlet_learnable_fused"}

# Default locations of the inference-engine bundle catalogs, relative to
# the parent of this repo. Customers and dev both place crl-train and
# inference-engine as siblings under one parent dir. There are two
# catalogs now — one per pod kind.
_INFERENCE_ENGINE = Path(__file__).resolve().parent.parent / "inference-engine"
_DEFAULT_DETECT_BUNDLES_PARENT = _INFERENCE_ENGINE / "detect-bundles"
_DEFAULT_CLASSIFY_BUNDLES_PARENT = _INFERENCE_ENGINE / "classify-bundles"

# Bundle names follow the convention documented in each catalog's README:
#   detect:   <frontend>-<training-mode>-<run-id>-v<N>
#   classify: <frontend>-<training-mode>-<run-id>-<probe>-v<N>
# We don't enforce the full grammar — just that it ends in -v<N> so a
# version exists, and that there's no path separator (so a typo can't
# write outside the catalog dir).
_BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*-v\d+$")

# Tie band for default-symlink promotion. Two runs are considered tied
# iff |primary_a - primary_b| < _TIE_EPSILON; tie-breaker fires inside
# the band only.
_TIE_EPSILON = 0.01

# Floors for *-default symlink promotion. Floors gate ONLY auto-promotion;
# bundles below the floor still exist on disk and can be selected
# explicitly via DETECT_BUNDLE / CLASSIFY_BUNDLE.
_PROMOTION_FLOOR = {
    "detect": ("pres_f1", 0.80),
    "classify": ("min_type_f1", 0.40),
}

# (primary_metric, tiebreaker_metric) per kind.
_RANKING_METRICS = {
    "detect": ("pres_f1", "min_pres_f1"),
    "classify": ("type_f1", "min_type_f1"),
}


# ---------------------------------------------------------------------------
# Morlet bank wrappers — resolve the FFT-vs-conv branch at export time so
# TorchScript never sees the class-level constant comparison.
# ---------------------------------------------------------------------------

import math

import torch.nn.functional as F


class _MorletPostprocess(nn.Module):
    """Shared post-conv tail: log_power (+ phase) -> output channels."""

    def __init__(self, use_phase: bool) -> None:
        super().__init__()
        self.use_phase = use_phase

    def forward(self, re_out: torch.Tensor, im_out: torch.Tensor) -> torch.Tensor:
        if self.use_phase:
            mag = torch.sqrt(re_out.pow(2) + im_out.pow(2) + 1e-8)
            cos_phase = re_out / mag
            sin_phase = im_out / mag
            log_power = torch.log1p(re_out.pow(2) + im_out.pow(2))
            return torch.cat([log_power, cos_phase, sin_phase], dim=1)
        power = re_out.pow(2) + im_out.pow(2)
        return torch.log1p(power)


def _fft_apply(
    x: torch.Tensor, kernel_re: torch.Tensor, kernel_im: torch.Tensor, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared FFT cross-correlation. Free function so both fixed and
    learnable FFT classes can call it without TorchScript inheritance
    quirks."""
    L = x.shape[-1]
    ks = kernel_size
    n_fft = 1
    target = L + ks - 1
    while n_fft < target:
        n_fft <<= 1
    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    K_re = torch.fft.rfft(kernel_re, n=n_fft, dim=-1)
    K_im = torch.fft.rfft(kernel_im, n=n_fft, dim=-1)
    Y_re = torch.einsum("bif,oif->bof", X, K_re.conj())
    Y_im = torch.einsum("bif,oif->bof", X, K_im.conj())
    re_full = torch.fft.irfft(Y_re, n=n_fft, dim=-1)
    im_full = torch.fft.irfft(Y_im, n=n_fft, dim=-1)
    shift = ks // 2
    re_out = torch.roll(re_full, shifts=shift, dims=-1)[..., :L]
    im_out = torch.roll(im_full, shifts=shift, dims=-1)[..., :L]
    return re_out, im_out


def _build_learnable_kernels(
    log_scales: torch.Tensor,
    w0_per_filter: torch.Tensor | None,
    w0_scalar: float,
    kernel_size: int,
    sample_rate: int,
    in_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = log_scales.exp()
    t = torch.linspace(
        -kernel_size // 2,
        kernel_size // 2,
        kernel_size,
        device=scales.device,
    ).float() / float(sample_rate)
    s = scales.unsqueeze(-1)
    if w0_per_filter is not None:
        w0 = w0_per_filter.unsqueeze(-1)
    else:
        w0 = torch.full_like(s, w0_scalar)
    norm = (math.pi * s) ** -0.25
    gauss = torch.exp(-0.5 * (t / s) ** 2)
    kernel_re = (norm * gauss * torch.cos(w0 * t / s)).unsqueeze(1).expand(-1, in_channels, -1)
    kernel_im = (norm * gauss * torch.sin(w0 * t / s)).unsqueeze(1).expand(-1, in_channels, -1)
    return kernel_re, kernel_im


class _FixedConvBank(nn.Module):
    """Fixed-Morlet kernels, direct-conv path. Used when kernel_size <
    FFT_CONV_THRESHOLD AND bank is not Learnable."""

    def __init__(self, bank: nn.Module) -> None:
        super().__init__()
        self.padding = int(bank.padding)
        self.register_buffer("kernel_re", bank.kernel_re)
        self.register_buffer("kernel_im", bank.kernel_im)
        self.post = _MorletPostprocess(use_phase=bool(bank.use_phase))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        re_out = F.conv1d(x, self.kernel_re, padding=self.padding)
        im_out = F.conv1d(x, self.kernel_im, padding=self.padding)
        return self.post(re_out, im_out)


class _FixedFFTBank(nn.Module):
    """Fixed-Morlet kernels, FFT path. Used when kernel_size >=
    FFT_CONV_THRESHOLD AND bank is not Learnable."""

    def __init__(self, bank: nn.Module) -> None:
        super().__init__()
        self.kernel_size = int(bank.kernel_size)
        self.register_buffer("kernel_re", bank.kernel_re)
        self.register_buffer("kernel_im", bank.kernel_im)
        self.post = _MorletPostprocess(use_phase=bool(bank.use_phase))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        re_out, im_out = _fft_apply(x, self.kernel_re, self.kernel_im, self.kernel_size)
        return self.post(re_out, im_out)


class _LearnableConvBank(nn.Module):
    """Learnable-Morlet kernels (rebuilt each forward), direct-conv path."""

    def __init__(self, bank: nn.Module) -> None:
        super().__init__()
        self.padding = int(bank.padding)
        self.kernel_size = int(bank.kernel_size)
        self.in_channels = int(bank.in_channels)
        self.sample_rate = int(bank.sample_rate)
        self.log_scales = bank.log_scales
        self.learnable_w0 = bool(bank.learnable_w0)
        if self.learnable_w0:
            self.w0_per_filter = bank.w0_per_filter
            self.w0_scalar = 0.0
        else:
            self.register_parameter("w0_per_filter", None)
            self.w0_scalar = float(bank.w0)
        self.post = _MorletPostprocess(use_phase=bool(bank.use_phase))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        kernel_re, kernel_im = _build_learnable_kernels(
            self.log_scales,
            self.w0_per_filter,
            self.w0_scalar,
            self.kernel_size,
            self.sample_rate,
            self.in_channels,
        )
        re_out = F.conv1d(x, kernel_re, padding=self.padding)
        im_out = F.conv1d(x, kernel_im, padding=self.padding)
        return self.post(re_out, im_out)


class _LearnableFFTBank(nn.Module):
    """Learnable-Morlet kernels (rebuilt each forward), FFT path."""

    def __init__(self, bank: nn.Module) -> None:
        super().__init__()
        self.kernel_size = int(bank.kernel_size)
        self.in_channels = int(bank.in_channels)
        self.sample_rate = int(bank.sample_rate)
        self.log_scales = bank.log_scales
        self.learnable_w0 = bool(bank.learnable_w0)
        if self.learnable_w0:
            self.w0_per_filter = bank.w0_per_filter
            self.w0_scalar = 0.0
        else:
            self.register_parameter("w0_per_filter", None)
            self.w0_scalar = float(bank.w0)
        self.post = _MorletPostprocess(use_phase=bool(bank.use_phase))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        kernel_re, kernel_im = _build_learnable_kernels(
            self.log_scales,
            self.w0_per_filter,
            self.w0_scalar,
            self.kernel_size,
            self.sample_rate,
            self.in_channels,
        )
        re_out, im_out = _fft_apply(x, kernel_re, kernel_im, self.kernel_size)
        return self.post(re_out, im_out)


def _learnable_cls():
    """Lazy import to avoid pulling crl_vehicle into module-load if not needed."""
    from crl_vehicle.models.frontend import LearnableMorletFilterbank

    return LearnableMorletFilterbank


def _resolve_morlet_bank(bank: nn.Module) -> nn.Module:
    """Pick (FFT vs conv) × (fixed vs learnable) at export time and return
    a scriptable replacement. The chosen module's forward has no runtime
    branches on bank type or kernel size, so TorchScript compiles cleanly.
    """
    from crl_vehicle.models.frontend import MorletFilterbank

    fft_threshold = MorletFilterbank.FFT_CONV_THRESHOLD
    use_fft = int(bank.kernel_size) >= fft_threshold
    is_learnable = isinstance(bank, _learnable_cls())
    if is_learnable and use_fft:
        return _LearnableFFTBank(bank)
    if is_learnable and not use_fft:
        return _LearnableConvBank(bank)
    if use_fft:
        return _FixedFFTBank(bank)
    return _FixedConvBank(bank)


def _resolve_morlet_in_frontend(seq: nn.Module) -> nn.Module:
    """Walk a frontend's children, swap any MorletFilterbank for its
    resolved equivalent. Returns a new nn.Sequential-like module."""
    from crl_vehicle.models.frontend import MorletFilterbank

    if not isinstance(seq, nn.Sequential):
        return seq
    new_children = []
    for child in seq:
        if isinstance(child, MorletFilterbank):
            new_children.append(_resolve_morlet_bank(child))
        else:
            new_children.append(child)
    return nn.Sequential(*new_children)


def _verify_morlet_resolution(
    original: nn.Module, resolved: nn.Module, window_size: int, atol: float = 1e-4
) -> None:
    """Sanity-check that the resolved (path-pinned) frontend produces the
    same output as the original frontend. Run before scripting so any
    resolution bug surfaces with a clear diff rather than as a parity
    failure later. atol is looser than the script-vs-resolve check because
    FFT vs direct-conv can differ at ~1e-5 due to floating-point order."""
    if not isinstance(original, nn.Sequential):
        return
    x = torch.randn(2, 1, window_size)
    with torch.no_grad():
        y0 = original(x)
        y1 = resolved(x)
    if y0.shape != y1.shape:
        raise RuntimeError(
            f"Morlet resolution shape mismatch: orig={tuple(y0.shape)} "
            f"resolved={tuple(y1.shape)}"
        )
    diff = (y0 - y1).abs().max().item()
    if diff > atol:
        raise RuntimeError(
            f"Morlet resolution numeric mismatch: max diff = {diff:.2e} " f"(atol={atol:.0e})"
        )


# ---------------------------------------------------------------------------
# Wrapper modules — flat, no string-keyed ModuleDict, scriptable
# ---------------------------------------------------------------------------


class EncoderPresencePerSensor(nn.Module):
    """One sensor's frontend → encoder → pres_head pipeline.

    Returns (z, pres_logit). z is the full latent (B, d_z) so the type pod
    can apply whichever slice it needs without coordinating with the encoder.
    """

    def __init__(
        self,
        frontend: nn.Module,
        encoder: nn.Module,
        pres_head: nn.Module,
        d_pres: int,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.pres_head = pres_head
        self.d_pres = d_pres

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.frontend(x.float())
        z, mu, logvar = self.encoder(feats)
        z_pres = z[:, : self.d_pres]
        pres_logit = self.pres_head(z_pres)
        return z, pres_logit


class EncoderPresenceFused(nn.Module):
    """Fused (early-fusion) frontends → shared encoder → pres_head.

    Frontends list order MUST match the sensor order used at training time
    (see CRLModel.encode_fused: zip(self.sensors, [x_audio, x_seismic])).
    The exporter passes [audio_frontend, seismic_frontend] to match.
    """

    def __init__(
        self,
        frontend_audio: nn.Module,
        frontend_seismic: nn.Module,
        encoder: nn.Module,
        pres_head: nn.Module,
        d_pres: int,
    ) -> None:
        super().__init__()
        self.frontend_audio = frontend_audio
        self.frontend_seismic = frontend_seismic
        self.encoder = encoder
        self.pres_head = pres_head
        self.d_pres = d_pres

    def forward(
        self, x_audio: torch.Tensor, x_seismic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_audio = self.frontend_audio(x_audio.float())
        f_seismic = self.frontend_seismic(x_seismic.float())
        features = torch.cat([f_audio, f_seismic], dim=2)
        z, mu, logvar = self.encoder(features)
        z_pres = z[:, : self.d_pres]
        pres_logit = self.pres_head(z_pres)
        return z, pres_logit


class TypeOnZ(nn.Module):
    """Apply a type head to the latent z.

    Slicing depends on probe_mode:
      linear_ztype / mlp_ztype     -> z[:, d_pres : d_pres + d_type]
      linear_fullz                 -> z (no slice)
      linear_signal / mlp_signal   -> z[:, 0 : d_signal]

    The exporter resolves the slice once at export time and stores
    (slice_start, slice_end) so the scripted module never branches on
    probe_mode at runtime.
    """

    def __init__(
        self,
        type_head: nn.Module,
        slice_start: int,
        slice_end: int,
    ) -> None:
        super().__init__()
        self.type_head = type_head
        self.slice_start = slice_start
        self.slice_end = slice_end

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_in = z[:, self.slice_start : self.slice_end]
        return self.type_head(z_in)


# ---------------------------------------------------------------------------
# Slicing math
# ---------------------------------------------------------------------------


def resolve_type_slice(probe_mode: str, cfg: CRLConfig) -> tuple[int, int]:
    """Return (start, end) indices into z for the type head's input."""
    if probe_mode == "linear_ztype" or probe_mode == "mlp_ztype":
        # CausalLatentSpace type slice = [D_PRES : D_PRES + D_TYPE].
        from crl_vehicle.models.latent import CausalLatentSpace

        start = CausalLatentSpace.D_PRES
        return start, start + CausalLatentSpace.D_TYPE
    if probe_mode == "linear_fullz":
        return 0, cfg.d_z
    if probe_mode == "linear_signal" or probe_mode == "mlp_signal":
        return 0, cfg.d_signal
    raise ValueError(f"Unknown probe_mode: {probe_mode!r}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _infer_multiscale_kernels_from_checkpoint(
    state: dict, sensors: list[str]
) -> dict[str, list[int]]:
    """Reconstruct multiscale_kernel_sizes from a saved state_dict.

    Older checkpoints don't store kernel-size overrides in meta.json, so the
    live CRLConfig default (which has changed over time) won't match the
    weight shapes. Read the actual branch count and kernel sizes off
    `frontends.{sensor}.0.branches.{i}.0.weight` and use them to override
    the config — guaranteeing CRLModel(cfg) builds modules whose state_dict
    keys line up with the checkpoint.
    """
    out: dict[str, list[int]] = {}
    for sensor in sensors:
        prefix = f"frontends.{sensor}.0.branches."
        branch_keys = sorted(
            int(k[len(prefix) :].split(".")[0])
            for k in state
            if k.startswith(prefix) and k.endswith(".0.weight")
        )
        if not branch_keys:
            continue
        kernels = []
        for i in branch_keys:
            w = state[f"{prefix}{i}.0.weight"]
            kernels.append(int(w.shape[-1]))
        out[sensor] = kernels
    return out


def _rebuild_morlet_frontends_from_meta(model: CRLModel, derived: dict, sensors: list[str]) -> None:
    """Replace the live morlet frontends with ones whose kernel_size and
    pool_stride match the values recorded at training time.

    Necessary because `_init_morlet_per_sensor` derives those values from
    `morlet_per_sensor_params` using a formula that has changed over time;
    a checkpoint trained against an older formula won't load against the
    live formula's shapes. The saved `morlet_derived_params` block records
    the actual values used during training — trust those.
    """
    from crl_vehicle.models.frontend import LearnableMorletFilterbank, MorletFilterbank

    cfg = model.cfg
    use_phase = cfg.morlet_use_phase
    is_learnable = cfg.frontend_bank == "morlet_learnable"

    for sensor in sensors:
        if sensor not in derived:
            continue
        d = derived[sensor]
        sp = cfg.frontend_per_sensor_params[sensor]
        mc = cfg.modality_cfg(sensor)
        out_channels = max(1, int(round(cfg.d_model * sp.get("out_channels_frac", 1.0))))

        bank_cls = LearnableMorletFilterbank if is_learnable else MorletFilterbank
        bank = bank_cls(
            in_channels=mc.n_channels,
            out_channels=out_channels,
            kernel_size=int(d["kernel_size"]),
            sample_rate=mc.sample_rate,
            w0=float(sp.get("w0", 6.0)),
            freq_min=float(sp["freq_min"]),
            freq_max=float(sp["freq_max"]),
            use_phase=use_phase,
        )
        pool_stride = int(d["pool_stride"])
        model.frontends[sensor] = nn.Sequential(
            bank,
            nn.AvgPool1d(pool_stride, pool_stride),
        )


def _infer_d_type_from_checkpoint(state: dict) -> int | None:
    """Read the trained ``D_TYPE`` from a saved CRL checkpoint.

    The aux_type head is a ``Linear(D_TYPE, 4)`` fed by ``z_type``, the
    type slice produced by ``CausalLatentSpace.split``. Reading the
    weight's input dim recovers the ``D_TYPE`` that was in effect when
    the run was trained — which has drifted in main-line code (e.g.
    6 ↔ 12 across configs). The downstream type-head also has this
    same input dim by construction (``MLPTypeHead()`` defaults to
    ``CausalLatentSpace.D_TYPE``), so a single override realigns the
    z-slice and the head shapes simultaneously.
    """
    for k in ("aux_type_heads.fused.weight",) + tuple(
        f"aux_type_heads.{s}.weight" for s in ("audio", "seismic")
    ):
        w = state.get(k)
        if w is not None:
            return int(w.shape[1])
    return None


def load_trained_model(save_dir: Path, encoder_from: str = "pres") -> tuple[CRLModel, dict]:
    """Load a trained CRLModel by merging the two head-specific checkpoints.

    Presence-side parameters (encoder, frontends, latent, pres_heads, aux_pres_heads)
    come from downstream_best_pres.pth. Type-side parameters (type_heads,
    aux_type_heads) come from downstream_best_type.pth. The shared encoder source
    is selected by ``encoder_from`` ('pres' or 'type'). For frozen-backbone runs
    both ckpts have bit-identical encoder state, so the default ('pres') is
    equivalent to either choice.
    """
    meta_path = save_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {save_dir}")
    meta = json.loads(meta_path.read_text())

    cfg_dict = meta.get("config", {})
    sensors = meta.get("sensors", ["audio", "seismic"])
    probe_mode = meta.get("probe_mode", "linear_ztype")

    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in CRLConfig.__dataclass_fields__}

    pres_path = save_dir / "downstream_best_pres.pth"
    type_path = save_dir / "downstream_best_type.pth"
    for p in (pres_path, type_path):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found in {save_dir}")

    pres_state = torch.load(pres_path, map_location="cpu", weights_only=True)
    type_state = torch.load(type_path, map_location="cpu", weights_only=True)

    # Build merged state_dict. Type-head keys come from type_state; presence-head
    # keys come from pres_state; everything else (encoder, frontends, latent
    # space) comes from whichever ckpt the user named via --encoder-from.
    encoder_state = pres_state if encoder_from == "pres" else type_state
    merged: dict = dict(encoder_state)
    for k, v in type_state.items():
        if k.startswith(("type_heads.", "aux_type_heads.")):
            merged[k] = v
    for k, v in pres_state.items():
        if k.startswith(("pres_heads.", "aux_pres_heads.")):
            merged[k] = v
    state = merged

    # Realign CausalLatentSpace.D_TYPE with the trained checkpoint before
    # building the model. Heads belong to the run, not the live code.
    from crl_vehicle.models.heads import LinearTypeHead, MLPTypeHead
    from crl_vehicle.models.latent import CausalLatentSpace

    trained_d_type = _infer_d_type_from_checkpoint(state)
    if trained_d_type is not None and trained_d_type != CausalLatentSpace.D_TYPE:
        print(
            f"  Overriding CausalLatentSpace.D_TYPE: "
            f"live={CausalLatentSpace.D_TYPE} -> trained={trained_d_type}"
        )
        CausalLatentSpace.D_TYPE = trained_d_type
        CausalLatentSpace.D_CAUSAL = (
            CausalLatentSpace.D_PRES
            + CausalLatentSpace.D_TYPE
            + CausalLatentSpace.D_PROX
            + CausalLatentSpace.D_ENV
        )
        # The type-head classes captured the old D_TYPE in their default
        # args at module-import time. Rebind those defaults so that
        # ``MLPTypeHead()`` / ``LinearTypeHead()`` (called with no args
        # by trainer._build_type_head) construct heads with the trained
        # input dim. aux_type_heads are built inline as
        # ``nn.Linear(CausalLatentSpace.D_TYPE, 4)``, so the class-attr
        # patch above already covers them.
        for cls in (LinearTypeHead, MLPTypeHead):
            sig_defaults = list(cls.__init__.__defaults__ or ())
            if sig_defaults:
                sig_defaults[0] = trained_d_type
                cls.__init__.__defaults__ = tuple(sig_defaults)

    frontend_type = cfg_dict.get("frontend_type", "multiscale")

    # Older checkpoints don't store the per-sensor kernel sizes in meta.json.
    # The live CRLConfig default has shifted (3 → 4 branches in commit
    # b9fb0ef), so reconstruct the actual sizes from the state_dict before
    # instantiating the model. The model reads kernel_sizes off
    # ``frontend_per_sensor_params[sensor]['kernel_sizes']`` (new schema),
    # not the legacy ``multiscale_kernel_sizes`` dict — overriding the
    # legacy field had no effect, leaving the model rebuilt with the live
    # default branch count and triggering a state_dict shape mismatch on
    # ``frontends.{sensor}.0.proj.weight``.
    if frontend_type == "multiscale":
        inferred = _infer_multiscale_kernels_from_checkpoint(state, sensors)
        if inferred:
            # Start from CRLConfig's live default for any sensor the saved
            # meta.json doesn't override -- otherwise we'd drop required keys
            # like ``target_tokens`` and ``out_channels_frac``.
            default_params = CRLConfig().frontend_per_sensor_params
            existing = cfg_kwargs.get("frontend_per_sensor_params") or {}
            params: dict[str, dict] = {s: dict(default_params.get(s, {})) for s in inferred}
            for s, p in existing.items():
                params.setdefault(s, {}).update(p)
            for sensor, ks in inferred.items():
                params[sensor]["kernel_sizes"] = ks
            cfg_kwargs["frontend_per_sensor_params"] = params
            cfg_kwargs.setdefault("multiscale_kernel_sizes", inferred)
            print(f"  Inferred multiscale kernel_sizes from checkpoint: {inferred}")

    cfg = CRLConfig(**cfg_kwargs)

    model = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode)

    # For morlet variants, the derived kernel_size / pool_stride formula has
    # changed over time. If the saved meta records `morlet_derived_params`,
    # rebuild the frontends to match the trained shapes BEFORE loading state.
    derived = meta.get("morlet_derived_params") or {}
    if derived and frontend_type in ("morlet_per_sensor", "morlet_learnable"):
        _rebuild_morlet_frontends_from_meta(model, derived, sensors)
        print("  Rebuilt morlet frontends from saved morlet_derived_params")

    model.load_state_dict(state)
    model.eval()
    return model, meta


# ---------------------------------------------------------------------------
# Wrapper construction (per mode)
# ---------------------------------------------------------------------------


def build_per_sensor_wrappers(
    model: CRLModel,
    sensor: str,
    type_slice: tuple[int, int],
) -> tuple[EncoderPresencePerSensor, TypeOnZ]:
    original_frontend = model.frontends[sensor]
    resolved_frontend = _resolve_morlet_in_frontend(original_frontend)
    if resolved_frontend is not original_frontend:
        window_size = model.cfg.modality_cfg(sensor).window_size
        _verify_morlet_resolution(original_frontend, resolved_frontend, window_size)
    enc_pres = EncoderPresencePerSensor(
        frontend=resolved_frontend,
        encoder=model.encoders[sensor],
        pres_head=model.pres_heads[sensor],
        d_pres=4,
    )
    type_on_z = TypeOnZ(
        type_head=model.type_heads[sensor],
        slice_start=type_slice[0],
        slice_end=type_slice[1],
    )
    return enc_pres.eval(), type_on_z.eval()


def build_fused_wrappers(
    model: CRLModel,
    type_slice: tuple[int, int],
) -> tuple[EncoderPresenceFused, TypeOnZ]:
    sensors = list(model.sensors)
    if sensors != ["audio", "seismic"]:
        raise NotImplementedError(
            f"Fused export currently assumes sensor order ['audio', 'seismic']; "
            f"got {sensors}. Update EncoderPresenceFused if you change the order."
        )
    for s in sensors:
        original = model.frontends[s]
        resolved = _resolve_morlet_in_frontend(original)
        if resolved is not original:
            _verify_morlet_resolution(
                original,
                resolved,
                window_size=model.cfg.modality_cfg(s).window_size,
            )
    enc_pres = EncoderPresenceFused(
        frontend_audio=_resolve_morlet_in_frontend(model.frontends["audio"]),
        frontend_seismic=_resolve_morlet_in_frontend(model.frontends["seismic"]),
        encoder=model.encoder,
        pres_head=model.pres_heads["fused"],
        d_pres=4,
    )
    type_on_z = TypeOnZ(
        type_head=model.type_heads["fused"],
        slice_start=type_slice[0],
        slice_end=type_slice[1],
    )
    return enc_pres.eval(), type_on_z.eval()


# ---------------------------------------------------------------------------
# Scripting + parity check
# ---------------------------------------------------------------------------


def script_and_save(module: nn.Module, path: Path) -> torch.jit.ScriptModule:
    scripted = torch.jit.script(module)
    scripted.save(str(path))
    return scripted


def parity_check_per_sensor(
    eager: EncoderPresencePerSensor,
    type_on_z_eager: TypeOnZ,
    scripted_enc: torch.jit.ScriptModule,
    scripted_type: torch.jit.ScriptModule,
    window_size: int,
    atol: float = 1e-5,
) -> None:
    x = torch.randn(2, 1, window_size)
    with torch.no_grad():
        z_e, p_e = eager(x)
        z_s, p_s = scripted_enc(x)
        t_e = type_on_z_eager(z_e)
        t_s = scripted_type(z_s)
    if not torch.allclose(z_e, z_s, atol=atol):
        raise RuntimeError(f"z parity failed: max diff = {(z_e - z_s).abs().max().item():.2e}")
    if not torch.allclose(p_e, p_s, atol=atol):
        raise RuntimeError(
            f"pres_logit parity failed: max diff = {(p_e - p_s).abs().max().item():.2e}"
        )
    if not torch.allclose(t_e, t_s, atol=atol):
        raise RuntimeError(
            f"type_logits parity failed: max diff = {(t_e - t_s).abs().max().item():.2e}"
        )


def parity_check_fused(
    eager: EncoderPresenceFused,
    type_on_z_eager: TypeOnZ,
    scripted_enc: torch.jit.ScriptModule,
    scripted_type: torch.jit.ScriptModule,
    audio_window: int,
    seismic_window: int,
    atol: float = 1e-5,
) -> None:
    x_a = torch.randn(2, 1, audio_window)
    x_s = torch.randn(2, 1, seismic_window)
    with torch.no_grad():
        z_e, p_e = eager(x_a, x_s)
        z_sc, p_sc = scripted_enc(x_a, x_s)
        t_e = type_on_z_eager(z_e)
        t_sc = scripted_type(z_sc)
    if not torch.allclose(z_e, z_sc, atol=atol):
        raise RuntimeError(
            f"fused z parity failed: max diff = {(z_e - z_sc).abs().max().item():.2e}"
        )
    if not torch.allclose(p_e, p_sc, atol=atol):
        raise RuntimeError(
            f"fused pres_logit parity failed: max diff = {(p_e - p_sc).abs().max().item():.2e}"
        )
    if not torch.allclose(t_e, t_sc, atol=atol):
        raise RuntimeError(
            f"fused type_logits parity failed: max diff = {(t_e - t_sc).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Deployment meta.json
# ---------------------------------------------------------------------------

CLASS_NAMES = ["pedestrian", "light", "medium", "heavy"]


def _find_run_root(save_dir: Path) -> Path:
    """Walk up from ``save_dir`` looking for the run root.

    The expected per-run layout is
    ``runs/<frontend>/<mode>/<run-id>/downstream/<probe>/`` with the
    checkpoints + meta.json in the probe dir, while the run root holds
    the downstream/, eval/, and crl/ subtrees that
    ``crl_vehicle.analysis`` needs to compute selection metrics. We
    detect the run root by the presence of a ``downstream/`` child.
    """
    cur = save_dir.resolve()
    # Probe dir -> downstream -> run -> mode -> frontend; cap at 6 levels.
    for _ in range(6):
        if (cur / "downstream").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        f"run root not found from {save_dir} (searched up to 6 levels for "
        f"a parent dir containing downstream/). Needed to compute bundle "
        f"selection metrics."
    )


def _min_or_none(values: dict[str, float]) -> float | None:
    if not values:
        return None
    return float(min(values.values()))


def _read_selection_metrics(save_dir: Path, kind: str) -> dict:
    """Compute the selection metrics the bundle catalog needs.

    Detect bundles use the run-level (canonical-probe) presence metrics:
    detect doesn't ship a probe, so per-bundle ranking is per-run. The
    canonical probe is whatever ``crl_vehicle.analysis.CANONICAL_PROBE``
    points at (``linear_ztype__crl_best`` today).

    Classify bundles use the *bundle's own* probe — ``save_dir`` is the
    probe directory the exporter was invoked against, so we use its
    basename as the probe name and read its downstream metrics + per-
    dataset eval directly. This lets two probes of the same run
    (``mlp_ztype`` vs ``linear_fullz``) compete as separate classify
    bundles.

    Missing required fields are a hard error so a bundle can't end up
    in the catalog without provenance. ``min_*`` (cross-location) is
    treated as required: per the design spec it's the ship-metric
    tie-breaker (and on classify, the promotion floor).
    """
    # Resolve the run root by walking up from the probe dir.
    run_root = _find_run_root(save_dir)

    # Pick which probe to read metrics from.
    if kind == "detect":
        probe_name = analysis.CANONICAL_PROBE
    elif kind == "classify":
        # ``save_dir`` IS the probe dir; its name is the probe id.
        probe_name = save_dir.name
    else:
        raise ValueError(f"unknown bundle kind: {kind!r}")

    # Pull val-split (downstream-loop) metrics from this probe's CSV.
    probe_dir = run_root / "downstream" / probe_name
    if not probe_dir.is_dir():
        raise FileNotFoundError(
            f"probe directory not found: {probe_dir}. "
            f"For kind={kind}, expected probe={probe_name!r} under "
            f"{run_root / 'downstream'}/."
        )
    ds_csv = probe_dir / "downstream_metrics.csv"
    if not ds_csv.exists():
        raise FileNotFoundError(
            f"{ds_csv} not found — needed to read val_pres_f1 / val_type_f1."
        )
    cols = analysis._read_csv_columns(ds_csv)
    best = analysis._best_from_ds_metrics(cols)

    # Pull cross-location per-dataset F1s from eval/<probe>/<head>/<split>/.
    cl = analysis._cross_location_metrics(run_root, preferred=probe_name)
    per_pres = cl.get("per_dataset_pres_f1") or {}
    per_type = cl.get("per_dataset_type_f1") or {}

    out: dict = {}
    if kind == "detect":
        pres_f1 = best.get("val_pres_f1")
        min_pres_f1 = _min_or_none(per_pres)
        if pres_f1 is None:
            raise KeyError(
                f"val_pres_f1 missing from {ds_csv}. Re-run downstream "
                f"training to produce it."
            )
        if min_pres_f1 is None:
            raise KeyError(
                f"no per-dataset presence F1s under "
                f"{run_root / 'eval' / probe_name / 'pres'}/. Re-run "
                f"run_full_diagnostic.py to produce per-location eval reports."
            )
        out["pres_f1"] = float(pres_f1)
        out["min_pres_f1"] = float(min_pres_f1)
    else:  # classify
        type_f1 = best.get("val_type_f1")
        min_type_f1 = _min_or_none(per_type)
        if type_f1 is None:
            raise KeyError(
                f"val_type_f1 missing from {ds_csv}. Re-run downstream "
                f"training to produce it."
            )
        if min_type_f1 is None:
            raise KeyError(
                f"no per-dataset type F1s under "
                f"{run_root / 'eval' / probe_name / 'type'}/. Re-run "
                f"run_full_diagnostic.py to produce per-location eval reports."
            )
        out["type_f1"] = float(type_f1)
        out["min_type_f1"] = float(min_type_f1)

    out["source_run"] = run_root.name
    return out


def build_deployment_meta(
    cfg: CRLConfig,
    sensors: list[str],
    mode: str,
    presence_threshold: dict | float,
    probe_mode: str,
    kind: str,
    selection_metrics: dict,
) -> dict:
    """Subset of training meta.json that the inference pods need.

    Inference pods only need: which mode, which sensors, what shapes the
    encoder expects, the threshold (detect-only), the class names
    (classify-only), the probe (classify-only), the latent dim, and the
    selection metrics for catalog ranking.
    """
    meta: dict = {
        "frontend_type": cfg.frontend_type,
        "mode": mode,  # "per_sensor" | "fused"
        "sensors": sensors,
        "z_dim": cfg.d_z,
    }
    if kind == "detect":
        meta["presence_threshold"] = presence_threshold
    elif kind == "classify":
        meta["class_names"] = CLASS_NAMES
        meta["probe_mode"] = probe_mode
    else:
        raise ValueError(f"unknown bundle kind: {kind!r}")

    for sensor in sensors:
        mc = cfg.modality_cfg(sensor)
        meta[f"{sensor}_sample_rate"] = mc.sample_rate
        meta[f"{sensor}_window_size"] = mc.window_size

    meta.update(selection_metrics)
    return meta


# ---------------------------------------------------------------------------
# Default-symlink promotion
# ---------------------------------------------------------------------------


def _list_catalog(bundles_dir: Path) -> list[Path]:
    """Return every bundle subdir in bundles_dir, ignoring symlinks
    (the *-default symlink itself is one of those) and non-dirs."""
    out: list[Path] = []
    for child in sorted(bundles_dir.iterdir()):
        if child.is_dir() and not child.is_symlink():
            out.append(child)
    return out


def _rank_bundles(catalog: list[Path], kind: str) -> list[tuple[Path, dict]]:
    """Rank catalog with the spec's ε-tie-band semantics.

    Two bundles are tied iff ``|primary_a - primary_b| < _TIE_EPSILON``
    (i.e., 0.01). Inside the band the tiebreaker fires; outside it the
    primary alone decides.

    Concretely: walk bundles in primary-descending order, building tie
    groups whose primary values are all within ε of the group's first
    member. Within each group sort by tiebreaker descending (then name
    ascending for determinism). Concatenate groups top-down.

    Bundles missing metrics are skipped with a warning.
    """
    primary, tiebreaker = _RANKING_METRICS[kind]
    scored: list[tuple[float, float, str, Path, dict]] = []
    for bundle in catalog:
        meta_path = bundle / "meta.json"
        if not meta_path.exists():
            print(f"  skipping {bundle.name}: no meta.json", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        if primary not in meta or tiebreaker not in meta:
            print(
                f"  skipping {bundle.name}: missing {primary!r} or {tiebreaker!r}",
                flush=True,
            )
            continue
        scored.append((float(meta[primary]), float(meta[tiebreaker]), bundle.name, bundle, meta))

    # Outer pass: sort by primary descending so we walk top-down.
    scored.sort(key=lambda t: (-t[0], t[2]))

    # Bin by ε around each group's anchor (the highest-primary unbinned
    # bundle). Within a bin, sort by (tiebreaker desc, name asc).
    out: list[tuple[Path, dict]] = []
    i = 0
    while i < len(scored):
        anchor_primary = scored[i][0]
        bin_end = i
        while (
            bin_end < len(scored)
            and (anchor_primary - scored[bin_end][0]) < _TIE_EPSILON
        ):
            bin_end += 1
        group = scored[i:bin_end]
        group.sort(key=lambda t: (-t[1], t[2]))
        out.extend((b, m) for _, _, _, b, m in group)
        i = bin_end
    return out


def _promote_default(bundles_dir: Path, kind: str) -> None:
    """Re-evaluate the catalog and repoint <kind>-default at the winner
    if any bundle clears the floor. Exits non-zero if no candidate is
    eligible — the just-written bundle still exists on disk and can be
    selected explicitly via DETECT_BUNDLE / CLASSIFY_BUNDLE.
    """
    floor_metric, floor_value = _PROMOTION_FLOOR[kind]
    primary, _tiebreaker = _RANKING_METRICS[kind]
    link_name = f"{kind}-default"
    link_path = bundles_dir / link_name

    catalog = _list_catalog(bundles_dir)
    if not catalog:
        print(f"  no bundles in {bundles_dir}; cannot promote {link_name}", flush=True)
        raise SystemExit(1)

    ranked = _rank_bundles(catalog, kind)
    if not ranked:
        print(f"  no bundles with valid metrics in {bundles_dir}", flush=True)
        raise SystemExit(1)

    eligible = [(b, m) for b, m in ranked if m.get(floor_metric, 0.0) >= floor_value]
    if not eligible:
        best_bundle, best_meta = ranked[0]
        print(
            f"  no eligible bundle for {link_name} "
            f"(highest {floor_metric}={best_meta.get(floor_metric, 0.0):.3f}, "
            f"floor={floor_value:.2f})",
            flush=True,
        )
        raise SystemExit(1)

    winner_bundle, winner_meta = eligible[0]
    target = winner_bundle.name

    current_target: str | None = None
    if link_path.is_symlink():
        current_target = os.readlink(link_path)

    if current_target == target:
        print(
            f"  {link_name} already points at {target} "
            f"({primary}={winner_meta[primary]:.3f}); no change",
            flush=True,
        )
        return

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    print(
        f"  repointed {link_path} -> {target} "
        f"({primary}={winner_meta[primary]:.3f})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--save-dir",
        required=True,
        type=Path,
        help="Saved-run directory (contains meta.json + downstream_best_pres.pth "
        "+ downstream_best_type.pth + report.json).",
    )
    ap.add_argument(
        "--bundle-kind",
        choices=["detect", "classify"],
        required=True,
        help=(
            "Which kind of bundle to write. "
            "'detect' = encoder + presence-only meta (deployed to infer-detect). "
            "'classify' = encoder + type head + classify meta (deployed to "
            "infer-classify). A single saved run produces one bundle per kind "
            "via two separate invocations."
        ),
    )

    target_group = ap.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--bundle-name",
        type=str,
        help=(
            "Name of a bundle directory under the per-kind catalog "
            "(detect-bundles/ or classify-bundles/, picked by --bundle-kind). "
            "Convention: "
            "  detect:   <frontend>-<training-mode>-<run-id>-v<N>. "
            "  classify: <frontend>-<training-mode>-<run-id>-<probe>-v<N>. "
            "Example: multiscale-vae-v3_lowfreq-mlp_ztype-v2"
        ),
    )
    target_group.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Explicit output directory (escape hatch for one-off exports). "
            "Use --bundle-name for the customer-facing path."
        ),
    )
    ap.add_argument(
        "--bundles-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory holding bundle subdirs. Defaults to "
            "<inference-engine>/<kind>-bundles/ per --bundle-kind."
        ),
    )
    ap.add_argument(
        "--promote-default",
        action="store_true",
        help=(
            "After a successful --bundle-name export, walk the catalog for "
            "this --bundle-kind, apply the selection rules, and repoint "
            "<kind>-default at the winner if it changed. Only valid with "
            "--bundle-name."
        ),
    )
    ap.add_argument(
        "--encoder-from",
        choices=["pres", "type"],
        default="pres",
        help="Which checkpoint supplies the shared encoder/frontend/latent "
        "weights (default: pres). For frozen-backbone runs the choice is "
        "irrelevant — the encoder state is bit-identical in both ckpts. Only "
        "matters when the run was trained with finetune_top_n != 0, in which "
        "case the two ckpts may have diverged encoder states.",
    )
    ap.add_argument(
        "--threshold-audio",
        type=float,
        default=0.5,
        help="Per-sensor mode: sigmoid threshold for audio presence (default: 0.5)",
    )
    ap.add_argument(
        "--threshold-seismic",
        type=float,
        default=0.5,
        help="Per-sensor mode: sigmoid threshold for seismic presence (default: 0.5)",
    )
    ap.add_argument(
        "--threshold-fused",
        type=float,
        default=0.5,
        help="Fused mode: sigmoid threshold for fused presence (default: 0.5)",
    )
    ap.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip parity check between eager and scripted modules",
    )

    args = ap.parse_args()

    # Resolve --bundles-dir per kind if not supplied.
    if args.bundles_dir is None:
        if args.bundle_kind == "detect":
            args.bundles_dir = _DEFAULT_DETECT_BUNDLES_PARENT
        else:
            args.bundles_dir = _DEFAULT_CLASSIFY_BUNDLES_PARENT

    # Resolve --bundle-name to a concrete out_dir, with validation.
    if args.bundle_name is not None:
        if not _BUNDLE_NAME_RE.match(args.bundle_name):
            ap.error(
                f"--bundle-name {args.bundle_name!r} doesn't match the convention "
                f"<name>-v<N>. See inference-engine/{args.bundle_kind}-bundles/"
                f"README.md for the full naming guide."
            )
        if not args.bundles_dir.is_dir():
            ap.error(
                f"--bundles-dir {args.bundles_dir} does not exist. "
                f"Pass --bundles-dir explicitly or place crl-train and "
                f"inference-engine as siblings under one parent dir."
            )
        args.out_dir = args.bundles_dir / args.bundle_name

    if args.promote_default and args.bundle_name is None:
        ap.error("--promote-default requires --bundle-name")

    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.save_dir} (encoder_from={args.encoder_from})")
    model, meta = load_trained_model(args.save_dir, encoder_from=args.encoder_from)
    cfg: CRLConfig = model.cfg
    sensors: list[str] = list(model.sensors)
    probe_mode: str = model.probe_mode
    print(f"  frontend_type={cfg.frontend_type}, sensors={sensors}, probe_mode={probe_mode}")

    type_slice = resolve_type_slice(probe_mode, cfg)
    print(f"  type_head reads z[:, {type_slice[0]}:{type_slice[1]}]")

    if cfg.frontend_type in PER_SENSOR_FRONTENDS:
        mode = "per_sensor"
        threshold_dict = {
            "audio": float(args.threshold_audio),
            "seismic": float(args.threshold_seismic),
        }
        presence_threshold: dict | float = threshold_dict

        for sensor in sensors:
            mc = cfg.modality_cfg(sensor)
            print(f"\nExporting per-sensor [{sensor}] (window={mc.window_size})")
            enc_eager, type_eager = build_per_sensor_wrappers(model, sensor, type_slice)
            enc_path = args.out_dir / f"encoder_{sensor}.ts"
            scripted_enc = script_and_save(enc_eager, enc_path)
            print(f"  wrote {enc_path.name}")

            scripted_type = None
            if args.bundle_kind == "classify":
                type_path = args.out_dir / f"type_head_{sensor}.ts"
                scripted_type = script_and_save(type_eager, type_path)
                print(f"  wrote {type_path.name}")

            if not args.skip_parity and scripted_type is not None:
                parity_check_per_sensor(
                    enc_eager,
                    type_eager,
                    scripted_enc,
                    scripted_type,
                    window_size=mc.window_size,
                )
                print("  parity OK (atol=1e-5)")

    elif cfg.frontend_type in FUSED_FRONTENDS:
        mode = "fused"
        presence_threshold = float(args.threshold_fused)

        print("\nExporting fused encoder (audio + seismic)")
        enc_eager, type_eager = build_fused_wrappers(model, type_slice)
        enc_path = args.out_dir / "encoder_fused.ts"
        scripted_enc = script_and_save(enc_eager, enc_path)
        print(f"  wrote {enc_path.name}")

        scripted_type = None
        if args.bundle_kind == "classify":
            type_path = args.out_dir / "type_head_fused.ts"
            scripted_type = script_and_save(type_eager, type_path)
            print(f"  wrote {type_path.name}")

        if not args.skip_parity and scripted_type is not None:
            audio_window = cfg.modality_cfg("audio").window_size
            seismic_window = cfg.modality_cfg("seismic").window_size
            parity_check_fused(
                enc_eager,
                type_eager,
                scripted_enc,
                scripted_type,
                audio_window=audio_window,
                seismic_window=seismic_window,
            )
            print("  parity OK (atol=1e-5)")

    else:
        raise NotImplementedError(
            f"Export not supported for frontend_type={cfg.frontend_type!r}. "
            f"Supported: per-sensor {sorted(PER_SENSOR_FRONTENDS)}, "
            f"fused {sorted(FUSED_FRONTENDS)}."
        )

    selection_metrics = _read_selection_metrics(args.save_dir, args.bundle_kind)

    deploy_meta = build_deployment_meta(
        cfg=cfg,
        sensors=sensors,
        mode=mode,
        presence_threshold=presence_threshold,
        probe_mode=probe_mode,
        kind=args.bundle_kind,
        selection_metrics=selection_metrics,
    )
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(deploy_meta, indent=2) + "\n")
    print(f"\nWrote {meta_path}")
    print(f"Done. {args.out_dir}/ ready for inference-engine deploy.")

    if args.promote_default:
        _promote_default(args.bundles_dir, args.bundle_kind)
    elif args.bundle_name is not None:
        link_name = f"{args.bundle_kind}-default"
        print(
            f"\nTo evaluate this against the catalog and promote if it wins, "
            f"re-run with --promote-default. Or manually:\n"
            f"  ln -sfn {args.bundle_name} "
            f"{args.bundles_dir}/{link_name}"
        )


if __name__ == "__main__":
    main()
