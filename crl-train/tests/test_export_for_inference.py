"""End-to-end parity tests for export_for_inference.py.

Validates that scripted artifacts produce the same outputs as the original
CRLModel forward path on synthetic windows, for all currently supported
frontend types. These tests run with small CRLConfig values so they fit in
unit-test time without saved checkpoints — they construct fresh models and
export them in-process.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel

from export_for_inference import (
    PER_SENSOR_FRONTENDS,
    FUSED_FRONTENDS,
    build_per_sensor_wrappers,
    build_fused_wrappers,
    build_deployment_meta,
    parity_check_per_sensor,
    parity_check_fused,
    resolve_type_slice,
    script_and_save,
)


def _small_multiscale_cfg() -> CRLConfig:
    # Small enough to run fast; mirrors the production frontend type.
    return CRLConfig(
        d_model=32, n_layers=1, n_heads=4,
        frontend_type="multiscale", fused_seq_len=8, d_z=32,
        multiscale_pool_stride=16,
    )


def _small_morlet_per_sensor_cfg(use_phase: bool) -> CRLConfig:
    # Small d_model + smaller per-sensor target_tokens to keep the kernel
    # size compact (so direct-conv path is exercised when use_phase=False
    # and FFT path when use_phase=True wouldn't change since both depend
    # on kernel_size only).
    return CRLConfig(
        d_model=16, n_layers=1, n_heads=4,
        frontend_type="morlet_per_sensor", d_z=32,
        morlet_use_phase=use_phase,
        morlet_per_sensor_params={
            "audio":   {"freq_min": 50.0, "freq_max": 8000.0,
                        "out_channels_frac": 1.0, "w0": 6.0,
                        "target_tokens": 16, "receptive_cycles": 2.0},
            "seismic": {"freq_min": 5.0,  "freq_max": 40.0,
                        "out_channels_frac": 1.0, "w0": 6.0,
                        "target_tokens": 16, "receptive_cycles": 2.0},
        },
    )


# ---------------------------------------------------------------------------
# Per-sensor (morlet_per_sensor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_phase", [False, True])
def test_per_sensor_export_parity(tmp_path: Path, use_phase: bool) -> None:
    cfg = _small_morlet_per_sensor_cfg(use_phase=use_phase)
    model = CRLModel(cfg, sensors=["audio", "seismic"], probe_mode="linear_ztype")
    model.eval()
    type_slice = resolve_type_slice("linear_ztype", cfg)

    for sensor in ["audio", "seismic"]:
        enc_eager, type_eager = build_per_sensor_wrappers(model, sensor, type_slice)
        enc_path = tmp_path / f"encoder_{sensor}.ts"
        type_path = tmp_path / f"type_head_{sensor}.ts"
        scripted_enc = script_and_save(enc_eager, enc_path)
        scripted_type = script_and_save(type_eager, type_path)

        window_size = cfg.modality_cfg(sensor).window_size
        parity_check_per_sensor(
            enc_eager, type_eager, scripted_enc, scripted_type,
            window_size=window_size,
        )

        # Reload as a fresh process would and verify shapes.
        reloaded_enc = torch.jit.load(str(enc_path))
        reloaded_type = torch.jit.load(str(type_path))
        x = torch.randn(2, 1, window_size)
        with torch.no_grad():
            z, pres_logit = reloaded_enc(x)
            type_logits = reloaded_type(z)
        assert z.shape == (2, cfg.d_z)
        assert pres_logit.shape == (2, 1)
        assert type_logits.shape == (2, 4)


# ---------------------------------------------------------------------------
# Fused (multiscale)
# ---------------------------------------------------------------------------

def test_fused_multiscale_export_parity(tmp_path: Path) -> None:
    cfg = _small_multiscale_cfg()
    model = CRLModel(cfg, sensors=["audio", "seismic"], probe_mode="linear_ztype")
    model.eval()
    type_slice = resolve_type_slice("linear_ztype", cfg)

    enc_eager, type_eager = build_fused_wrappers(model, type_slice)
    enc_path = tmp_path / "encoder_fused.ts"
    type_path = tmp_path / "type_head_fused.ts"
    scripted_enc = script_and_save(enc_eager, enc_path)
    scripted_type = script_and_save(type_eager, type_path)

    audio_window = cfg.modality_cfg("audio").window_size
    seismic_window = cfg.modality_cfg("seismic").window_size
    parity_check_fused(
        enc_eager, type_eager, scripted_enc, scripted_type,
        audio_window=audio_window, seismic_window=seismic_window,
    )

    # Reload and verify shapes.
    reloaded_enc = torch.jit.load(str(enc_path))
    reloaded_type = torch.jit.load(str(type_path))
    x_a = torch.randn(2, 1, audio_window)
    x_s = torch.randn(2, 1, seismic_window)
    with torch.no_grad():
        z, pres_logit = reloaded_enc(x_a, x_s)
        type_logits = reloaded_type(z)
    assert z.shape == (2, cfg.d_z)
    assert pres_logit.shape == (2, 1)
    assert type_logits.shape == (2, 4)


# ---------------------------------------------------------------------------
# Deployment meta.json
# ---------------------------------------------------------------------------

def test_deployment_meta_per_sensor_includes_dict_threshold() -> None:
    cfg = _small_morlet_per_sensor_cfg(use_phase=False)
    meta = build_deployment_meta(
        cfg=cfg, sensors=["audio", "seismic"], mode="per_sensor",
        presence_threshold={"audio": 0.4, "seismic": 0.6},
        probe_mode="linear_ztype",
    )
    assert meta["mode"] == "per_sensor"
    assert meta["frontend_type"] == "morlet_per_sensor"
    assert meta["sensors"] == ["audio", "seismic"]
    assert meta["class_names"] == ["pedestrian", "light", "medium", "heavy"]
    assert meta["presence_threshold"] == {"audio": 0.4, "seismic": 0.6}
    assert meta["audio_sample_rate"] == 16000
    assert meta["audio_window_size"] == 16000
    assert meta["seismic_sample_rate"] == 100
    assert meta["seismic_window_size"] == 100
    assert meta["z_dim"] == cfg.d_z


def test_deployment_meta_fused_includes_scalar_threshold() -> None:
    cfg = _small_multiscale_cfg()
    meta = build_deployment_meta(
        cfg=cfg, sensors=["audio", "seismic"], mode="fused",
        presence_threshold=0.55, probe_mode="linear_ztype",
    )
    assert meta["mode"] == "fused"
    assert meta["frontend_type"] == "multiscale"
    assert meta["presence_threshold"] == 0.55


# ---------------------------------------------------------------------------
# Slice math
# ---------------------------------------------------------------------------

def test_resolve_type_slice_modes() -> None:
    cfg = CRLConfig(d_z=32, d_signal=12)
    # Type slice = [D_PRES : D_PRES + D_TYPE] = [4 : 4 + 12] = (4, 16)
    assert resolve_type_slice("linear_ztype", cfg) == (4, 16)
    assert resolve_type_slice("mlp_ztype", cfg) == (4, 16)
    assert resolve_type_slice("mlp_signal", cfg) == (0, 12)
    assert resolve_type_slice("linear_fullz", cfg) == (0, 32)
    assert resolve_type_slice("linear_signal", cfg) == (0, 12)
    with pytest.raises(ValueError):
        resolve_type_slice("unknown_mode", cfg)


# ---------------------------------------------------------------------------
# Frontend coverage
# ---------------------------------------------------------------------------

def test_frontend_set_membership_is_disjoint() -> None:
    """Sanity check that no frontend type is in both sets — would cause
    ambiguous mode selection in main()."""
    assert PER_SENSOR_FRONTENDS.isdisjoint(FUSED_FRONTENDS)
