"""Unified frontend factory for CRLModel.

Collapses the 3 bank types × 2 fusion modes (= 6 historical `frontend_type`
strings) into one orchestration function. Each bank type has a small builder
that returns `(nn.Module, n_out_channels)`; `build_frontend` wires the bank
into the per-sensor stack, picks late-fusion (per-sensor encoders) or
early-fusion (shared encoder + time-concat) topology, and returns everything
the caller assembles into a CRLModel.

The bank position in each per-sensor `nn.Sequential` is pinned at index 0 so
`learnable_morlet_parameters()` discovery (in trainer.py) and stage-2
checkpoint loading (which keys on `frontends.{sensor}.0.kernel_re`) keep
working unchanged.
"""

from __future__ import annotations

import math

import torch.nn as nn

from crl_vehicle.config import CRLConfig, ModalityConfig
from crl_vehicle.models.encoder_decoder import FeatureDecoder, TemporalEncoder
from crl_vehicle.models.frontend import (
    LearnableMorletFilterbank,
    MorletFilterbank,
    MultiScale1DFrontend,
)


def build_frontend(
    config: CRLConfig,
    sensors: list[str],
) -> tuple[
    nn.ModuleDict,  # frontends (per-sensor Sequential, bank at index 0)
    nn.Module | None,  # shared encoder (early fusion) or None
    nn.Module | None,  # shared decoder (early fusion) or None
    nn.ModuleDict,  # per-sensor encoders (late fusion) or empty
    nn.ModuleDict,  # per-sensor decoders (late fusion) or empty
    dict[str, dict],  # _morlet_derived_params audit trail
]:
    bank = config.frontend_bank
    fusion = config.frontend_fusion
    use_phase = config.morlet_use_phase
    learnable_w0 = config.morlet_learnable_w0
    params_by_sensor = config.frontend_per_sensor_params

    if fusion not in ("late", "early"):
        raise ValueError(f"frontend_fusion must be 'late' or 'early', got {fusion!r}")
    if bank not in ("multiscale", "morlet", "morlet_learnable"):
        raise ValueError(
            f"frontend_bank must be 'multiscale', 'morlet', or 'morlet_learnable', " f"got {bank!r}"
        )

    frontends: nn.ModuleDict = nn.ModuleDict()
    encoders: nn.ModuleDict = nn.ModuleDict()
    decoders: nn.ModuleDict = nn.ModuleDict()
    derived: dict[str, dict] = {}

    per_sensor_n_channels: dict[str, int] = {}
    per_sensor_target_tokens: dict[str, int] = {}

    for sensor in sensors:
        if sensor not in params_by_sensor:
            raise ValueError(
                f"frontend_bank={bank!r} requires params for {sensor!r} in "
                f"config.frontend_per_sensor_params (got keys "
                f"{list(params_by_sensor.keys())})"
            )
        sp = params_by_sensor[sensor]
        mc = config.modality_cfg(sensor)
        target_tokens = int(sp["target_tokens"])

        if bank == "multiscale":
            stack, n_out = _build_multiscale_bank(sp, mc, config.d_model, target_tokens)
            # Multiscale leaves derived empty (matches legacy contract).
        elif bank == "morlet":
            stack, n_out, derived[sensor] = _build_morlet_bank(
                sp,
                mc,
                config.d_model,
                use_phase,
                target_tokens,
                fusion,
            )
        else:  # morlet_learnable
            stack, n_out, derived[sensor] = _build_morlet_learnable_bank(
                sp,
                mc,
                config.d_model,
                use_phase,
                learnable_w0,
                target_tokens,
                fusion,
            )

        frontends[sensor] = stack
        per_sensor_n_channels[sensor] = n_out
        per_sensor_target_tokens[sensor] = target_tokens

        if fusion == "late":
            encoders[sensor] = TemporalEncoder(
                in_channels=n_out,
                d_z=config.d_z,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
            )
            seq_len = target_tokens if bank == "multiscale" else derived[sensor]["post_pool_tokens"]
            decoders[sensor] = FeatureDecoder(
                out_channels=n_out,
                seq_len=max(1, seq_len),
                d_z=config.d_z,
                d_model=config.d_model,
            )

    if fusion == "early":
        shared = set(per_sensor_n_channels.values())
        if len(shared) > 1:
            raise ValueError(
                f"frontend_fusion='early' requires matching n_out_channels "
                f"across sensors (got {per_sensor_n_channels}); early fusion "
                f"concatenates along time and a channel mismatch breaks the concat."
            )
        n_out = next(iter(shared))
        encoder = TemporalEncoder(
            in_channels=n_out,
            d_z=config.d_z,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
        decoder = FeatureDecoder(
            out_channels=n_out,
            seq_len=sum(per_sensor_target_tokens.values()),
            d_z=config.d_z,
            d_model=config.d_model,
        )
    else:
        encoder = None
        decoder = None

    return frontends, encoder, decoder, encoders, decoders, derived


def _build_multiscale_bank(
    sp: dict,
    mc: ModalityConfig,
    d_model: int,
    target_tokens: int,
) -> tuple[nn.Module, int]:
    kernel_sizes = list(sp["kernel_sizes"])
    strides = sp.get("strides")
    if strides is not None:
        strides = list(strides)
    bank = MultiScale1DFrontend(
        in_channels=mc.n_channels,
        out_channels=d_model,
        kernel_sizes=kernel_sizes,
        strides=strides,
        target_tokens=target_tokens,
    )
    # Bank already aligns each branch to target_tokens via per-branch
    # AdaptiveAvgPool1d, so the per-sensor stack is just the bank wrapped
    # in a Sequential (keeps the index-0 bank-position contract).
    stack = nn.Sequential(bank)
    return stack, d_model


def _build_morlet_bank(
    sp: dict,
    mc: ModalityConfig,
    d_model: int,
    use_phase: bool,
    target_tokens: int,
    fusion: str,
) -> tuple[nn.Module, int, dict]:
    out_channels = max(1, int(round(d_model * sp.get("out_channels_frac", 1.0))))
    freq_min = float(sp["freq_min"])
    freq_max = float(sp["freq_max"])
    w0 = float(sp.get("w0", 6.0))
    receptive_cycles = float(sp.get("receptive_cycles", 3.0))

    pool_stride, kernel_size = _derive_morlet_kernel_and_stride(
        mc,
        freq_min,
        w0,
        target_tokens,
        receptive_cycles,
    )
    bank = MorletFilterbank(
        in_channels=mc.n_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        sample_rate=mc.sample_rate,
        w0=w0,
        freq_min=freq_min,
        freq_max=freq_max,
        use_phase=use_phase,
    )
    stack, derived = _wrap_morlet_bank_with_pooling(
        bank,
        pool_stride,
        target_tokens,
        fusion,
        mc,
        kernel_size,
        receptive_cycles,
    )
    return stack, bank.total_out_channels, derived


def _build_morlet_learnable_bank(
    sp: dict,
    mc: ModalityConfig,
    d_model: int,
    use_phase: bool,
    learnable_w0: bool,
    target_tokens: int,
    fusion: str,
) -> tuple[nn.Module, int, dict]:
    out_channels = max(1, int(round(d_model * sp.get("out_channels_frac", 1.0))))
    freq_min = float(sp["freq_min"])
    freq_max = float(sp["freq_max"])
    w0 = float(sp.get("w0", 6.0))
    receptive_cycles = float(sp.get("receptive_cycles", 3.0))

    pool_stride, kernel_size = _derive_morlet_kernel_and_stride(
        mc,
        freq_min,
        w0,
        target_tokens,
        receptive_cycles,
    )
    bank = LearnableMorletFilterbank(
        in_channels=mc.n_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        sample_rate=mc.sample_rate,
        w0=w0,
        freq_min=freq_min,
        freq_max=freq_max,
        use_phase=use_phase,
        learnable_w0=learnable_w0,
    )
    stack, derived = _wrap_morlet_bank_with_pooling(
        bank,
        pool_stride,
        target_tokens,
        fusion,
        mc,
        kernel_size,
        receptive_cycles,
    )
    derived["learnable"] = True
    derived["learnable_w0"] = learnable_w0
    return stack, bank.total_out_channels, derived


def _derive_morlet_kernel_and_stride(
    mc: ModalityConfig,
    freq_min: float,
    w0: float,
    target_tokens: int,
    receptive_cycles: float,
) -> tuple[int, int]:
    pool_stride = max(1, mc.window_size // target_tokens)
    ks_float = 2 * receptive_cycles * w0 / (2 * math.pi * freq_min) * mc.sample_rate
    kernel_size = max(3, int(round(ks_float)) | 1)  # odd, ≥3
    return pool_stride, kernel_size


def _wrap_morlet_bank_with_pooling(
    bank: nn.Module,
    pool_stride: int,
    target_tokens: int,
    fusion: str,
    mc: ModalityConfig,
    kernel_size: int,
    receptive_cycles: float,
) -> tuple[nn.Module, dict]:
    layers: list[nn.Module] = [bank, nn.AvgPool1d(pool_stride, pool_stride)]
    post_pool_tokens = mc.window_size // pool_stride
    derived = {
        "pool_stride": pool_stride,
        "kernel_size": kernel_size,
        "target_tokens": target_tokens,
        "receptive_cycles": receptive_cycles,
        "post_pool_tokens": post_pool_tokens,
        "post_pool_rate": round(mc.sample_rate / pool_stride, 3),
    }
    if fusion == "early":
        layers.append(nn.AdaptiveAvgPool1d(target_tokens))
        derived["adaptive_pool_T"] = target_tokens
    return nn.Sequential(*layers), derived
