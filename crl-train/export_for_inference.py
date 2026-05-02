"""Export a trained CRLModel into TorchScript artifacts for the
inference-engine deployment pipeline.

Reads a saved run (meta.json + downstream_best.pth) and emits per-mode
TorchScript files plus a deployment-only meta.json that the inference pods
consume. The inference image has zero Python dependency on crl_vehicle —
only torch.jit.load is needed.

Output layout (in --out-dir):

  Per-sensor mode (frontend_type ∈ {morlet, morlet_per_sensor, morlet_learnable}):
    encoder_audio.ts        # (x_audio[B,1,16000])    -> (z[B,d_z], pres_logit[B,1])
    encoder_seismic.ts      # (x_seismic[B,1,100])    -> (z[B,d_z], pres_logit[B,1])
    type_head_audio.ts      # (z[B,d_z])              -> type_logits[B,4]
    type_head_seismic.ts    # (z[B,d_z])              -> type_logits[B,4]
    meta.json

  Fused mode (frontend_type ∈ {multiscale, morlet_fused, morlet_learnable_fused}):
    encoder_fused.ts        # (x_audio[B,1,16000], x_seismic[B,1,100]) -> (z, pres_logit)
    type_head_fused.ts      # (z[B,d_z]) -> type_logits[B,4]
    meta.json

CLI:
    # Customer-facing path: write directly into the inference-engine
    # bundle catalog (assumes crl-train and inference-engine are
    # siblings under one parent dir).
    python export_for_inference.py \\
        --save-dir saved_crl/runs/multiscale/vae/<run>/downstream/<probe> \\
        --bundle-name multiscale-vae-<run>-<probe>-aux_type-v2

    # Promote the new bundle as the shipping default in the same run:
    python export_for_inference.py \\
        --save-dir saved_crl/runs/multiscale/vae/<run>/downstream/<probe> \\
        --bundle-name multiscale-vae-<run>-<probe>-aux_type-v2 \\
        --update-default-symlink

    # Escape hatch: explicit out-dir for one-off exports outside the
    # bundle catalog (e.g. ad-hoc evaluation).
    python export_for_inference.py --save-dir saved_crl/runs/<run> \\
        --out-dir /tmp/scratch-bundle
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel

PER_SENSOR_FRONTENDS = {"morlet", "morlet_per_sensor", "morlet_learnable"}
FUSED_FRONTENDS = {"multiscale", "morlet_fused", "morlet_learnable_fused"}

# Default location of the inference-engine bundle catalog, relative to
# the parent of this repo. Customers and dev both place crl-train and
# inference-engine as siblings under one parent dir.
_DEFAULT_BUNDLES_PARENT = Path(__file__).resolve().parent.parent / "inference-engine" / "crl-bundles"

# Bundle names follow the convention documented in
# inference-engine/crl-bundles/README.md:
#   <frontend>-<training-mode>-<run-id>-<probe>-[aux_type-]v<N>
# We don't enforce the full grammar — just that it ends in -v<N> so a
# version exists, and that there's no path separator (so a typo can't
# write outside crl-bundles/).
_BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*-v\d+$")


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


def load_trained_model(save_dir: Path, ckpt_name: str) -> tuple[CRLModel, dict]:
    """Load a trained CRLModel from a saved-run directory."""
    meta_path = save_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {save_dir}")
    meta = json.loads(meta_path.read_text())

    cfg_dict = meta.get("config", {})
    sensors = meta.get("sensors", ["audio", "seismic"])
    probe_mode = meta.get("probe_mode", "linear_ztype")

    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in CRLConfig.__dataclass_fields__}

    ckpt_path = save_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_name} not found in {save_dir}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

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


def build_deployment_meta(
    cfg: CRLConfig,
    sensors: list[str],
    mode: str,
    presence_threshold: dict | float,
    probe_mode: str,
) -> dict:
    """Subset of training meta.json that the inference pods need.

    Inference pods only need: which mode, which sensors, what shapes the
    encoder expects, the threshold, the class names, and a couple of sanity-
    check values. Anything training-only (loss weights, optimizer hparams,
    dataset config) is omitted.
    """
    meta: dict = {
        "frontend_type": cfg.frontend_type,
        "mode": mode,  # "per_sensor" | "fused"
        "sensors": sensors,
        "probe_mode": probe_mode,
        "class_names": CLASS_NAMES,
        "presence_threshold": presence_threshold,
        "z_dim": cfg.d_z,
    }
    for sensor in sensors:
        mc = cfg.modality_cfg(sensor)
        meta[f"{sensor}_sample_rate"] = mc.sample_rate
        meta[f"{sensor}_window_size"] = mc.window_size
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--save-dir",
        required=True,
        type=Path,
        help="Saved-run directory (contains meta.json + downstream_best.pth)",
    )

    target_group = ap.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--bundle-name",
        type=str,
        help=(
            "Name of a bundle directory under inference-engine/crl-bundles/. "
            "Resolved relative to ../inference-engine/crl-bundles/ unless "
            "--bundles-dir is set. Convention: "
            "<frontend>-<training-mode>-<run-id>-<probe>-[aux_type-]v<N>. "
            "Example: multiscale-vae-v3_lowfreq-mlp_ztype-aux_type-v2"
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
        default=_DEFAULT_BUNDLES_PARENT,
        help=(
            f"Parent directory holding bundle subdirs. "
            f"Default: {_DEFAULT_BUNDLES_PARENT}"
        ),
    )
    ap.add_argument(
        "--update-default-symlink",
        action="store_true",
        help=(
            "After a successful --bundle-name export, repoint "
            "<bundles-dir>/multiscale-default at the new bundle. "
            "Only valid with --bundle-name."
        ),
    )
    ap.add_argument(
        "--ckpt-name",
        default="downstream_best.pth",
        help="Checkpoint filename inside --save-dir (default: downstream_best.pth)",
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

    # Resolve --bundle-name to a concrete out_dir, with validation.
    if args.bundle_name is not None:
        if not _BUNDLE_NAME_RE.match(args.bundle_name):
            ap.error(
                f"--bundle-name {args.bundle_name!r} doesn't match the convention "
                f"<name>-v<N>. See inference-engine/crl-bundles/README.md for the "
                f"full naming guide."
            )
        if not args.bundles_dir.is_dir():
            ap.error(
                f"--bundles-dir {args.bundles_dir} does not exist. "
                f"Pass --bundles-dir explicitly or place crl-train and "
                f"inference-engine as siblings under one parent dir."
            )
        args.out_dir = args.bundles_dir / args.bundle_name

    if args.update_default_symlink and args.bundle_name is None:
        ap.error("--update-default-symlink requires --bundle-name")

    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.save_dir} (ckpt={args.ckpt_name})")
    model, meta = load_trained_model(args.save_dir, args.ckpt_name)
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
            type_path = args.out_dir / f"type_head_{sensor}.ts"
            scripted_enc = script_and_save(enc_eager, enc_path)
            scripted_type = script_and_save(type_eager, type_path)
            print(f"  wrote {enc_path.name} and {type_path.name}")
            if not args.skip_parity:
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
        type_path = args.out_dir / "type_head_fused.ts"
        scripted_enc = script_and_save(enc_eager, enc_path)
        scripted_type = script_and_save(type_eager, type_path)
        print(f"  wrote {enc_path.name} and {type_path.name}")
        if not args.skip_parity:
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

    deploy_meta = build_deployment_meta(
        cfg=cfg,
        sensors=sensors,
        mode=mode,
        presence_threshold=presence_threshold,
        probe_mode=probe_mode,
    )
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(deploy_meta, indent=2) + "\n")
    print(f"\nWrote {meta_path}")
    print(f"Done. {args.out_dir}/ ready for inference-engine deploy.")

    if args.update_default_symlink:
        # Repoint <bundles-dir>/multiscale-default at the new bundle.
        # Use a relative target so the symlink survives a repo move.
        link_path = args.bundles_dir / "multiscale-default"
        target = args.bundle_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target)
        print(f"Repointed {link_path} -> {target}")
        print(
            "Remember to update the catalog table in "
            "inference-engine/crl-bundles/README.md and commit."
        )
    elif args.bundle_name is not None:
        print(
            f"\nTo make this the new shipping default, re-run with "
            f"--update-default-symlink, or manually:\n"
            f"  ln -sfn {args.bundle_name} "
            f"{args.bundles_dir}/multiscale-default"
        )


if __name__ == "__main__":
    main()
