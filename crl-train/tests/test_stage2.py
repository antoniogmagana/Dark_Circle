"""Tests for two-stage CRL training machinery.

Covers:
  - find_compatible_run: strict-match selection, topology enforcement,
    no-match error, auto-picks newest compatible run.
  - CRLModel.load_from_fixed_morlet_checkpoint: encoder preserved,
    log_scales initialized from init_scales, probe heads dropped.
  - Trainer stage-2 LR groups and warmup-cosine schedule shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from crl_vehicle.config import CRLConfig
from crl_vehicle.stage2 import (
    _params_equal,
    find_compatible_run,
    resolve_source_checkpoint,
)
from training.trainer import CRLModel, Trainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_fake_run(
    root: Path,
    run_name: str,
    frontend: str,
    *,
    sensors=("audio", "seismic"),
    d_model: int = 64,
    morlet_use_phase: bool = True,
    morlet_per_sensor_params: dict | None = None,
) -> Path:
    """Create a minimal run dir with crl/meta.json + a fake crl_best.pth."""
    run_dir = root / run_name
    (run_dir / "crl").mkdir(parents=True, exist_ok=True)
    cfg = CRLConfig(
        frontend_type=frontend,
        d_model=d_model,
        morlet_use_phase=morlet_use_phase,
    )
    if morlet_per_sensor_params is not None:
        cfg.morlet_per_sensor_params = morlet_per_sensor_params
    meta = {
        "config": asdict(cfg),
        "sensors": list(sensors),
    }
    (run_dir / "crl" / "meta.json").write_text(json.dumps(meta))
    (run_dir / "crl" / "crl_best.pth").write_bytes(b"dummy")
    return run_dir


# ---------------------------------------------------------------------------
# find_compatible_run
# ---------------------------------------------------------------------------


class TestFindCompatibleRun:
    def _target_cfg(self, **overrides) -> dict:
        cfg = CRLConfig(
            frontend_type="morlet_learnable",
            d_model=64,
            morlet_use_phase=True,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return asdict(cfg)

    def test_picks_newest_compatible(self, tmp_path):
        # Three runs: old+compatible, newer+mismatch (wrong d_model), newest+compatible.
        old = _write_fake_run(tmp_path, "2026-04-01", "morlet_per_sensor")
        old.stat()  # ensure path exists
        mid = _write_fake_run(tmp_path, "2026-04-15", "morlet_per_sensor", d_model=32)
        newest = _write_fake_run(tmp_path, "2026-04-20", "morlet_per_sensor")

        # Force mtime order so newest actually has the latest mtime.
        import os
        import time

        os.utime(old / "crl" / "meta.json", (time.time() - 300, time.time() - 300))
        os.utime(mid / "crl" / "meta.json", (time.time() - 200, time.time() - 200))
        os.utime(newest / "crl" / "meta.json", (time.time() - 100, time.time() - 100))

        chosen = find_compatible_run(
            target_frontend="morlet_learnable",
            target_sensors=["audio", "seismic"],
            target_cfg=self._target_cfg(),
            search_root=tmp_path,
            verbose=False,
        )
        assert chosen == newest

    def test_topology_mismatch_rejected(self, tmp_path):
        """target=morlet_learnable (late fusion) must not pick morlet_fused source."""
        _write_fake_run(tmp_path, "only_run", "morlet_fused")
        with pytest.raises(RuntimeError, match="No compatible stage-1 run"):
            find_compatible_run(
                target_frontend="morlet_learnable",
                target_sensors=["audio", "seismic"],
                target_cfg=self._target_cfg(),
                search_root=tmp_path,
                verbose=False,
            )

    def test_fused_pairs_with_fused(self, tmp_path):
        """target=morlet_learnable_fused must pick morlet_fused, reject morlet_per_sensor."""
        _write_fake_run(tmp_path, "per_sensor_run", "morlet_per_sensor")
        fused_run = _write_fake_run(tmp_path, "fused_run", "morlet_fused")
        chosen = find_compatible_run(
            target_frontend="morlet_learnable_fused",
            target_sensors=["audio", "seismic"],
            target_cfg=self._target_cfg(frontend_type="morlet_learnable_fused"),
            search_root=tmp_path,
            verbose=False,
        )
        assert chosen == fused_run

    def test_phase_mismatch_rejected(self, tmp_path):
        _write_fake_run(tmp_path, "no_phase", "morlet_per_sensor", morlet_use_phase=False)
        with pytest.raises(RuntimeError, match="No compatible"):
            find_compatible_run(
                target_frontend="morlet_learnable",
                target_sensors=["audio", "seismic"],
                target_cfg=self._target_cfg(morlet_use_phase=True),
                search_root=tmp_path,
                verbose=False,
            )

    def test_no_candidates_at_all_raises(self, tmp_path):
        """Empty search root raises FileNotFoundError (distinct from 'no match')."""
        with pytest.raises(FileNotFoundError):
            find_compatible_run(
                target_frontend="morlet_learnable",
                target_sensors=["audio", "seismic"],
                target_cfg=self._target_cfg(),
                search_root=tmp_path,
                verbose=False,
            )

    def test_error_message_lists_all_candidates(self, tmp_path):
        _write_fake_run(tmp_path, "bad_phase", "morlet_per_sensor", morlet_use_phase=False)
        _write_fake_run(tmp_path, "bad_dmodel", "morlet_per_sensor", d_model=32)
        with pytest.raises(RuntimeError) as exc_info:
            find_compatible_run(
                target_frontend="morlet_learnable",
                target_sensors=["audio", "seismic"],
                target_cfg=self._target_cfg(),
                search_root=tmp_path,
                verbose=False,
            )
        msg = str(exc_info.value)
        assert "bad_phase" in msg
        assert "bad_dmodel" in msg
        assert "morlet_use_phase" in msg
        assert "d_model" in msg


class TestResolveSourceCheckpoint:
    def test_prefers_crl_best(self, tmp_path):
        run = _write_fake_run(tmp_path, "run", "morlet_per_sensor")
        (run / "crl" / "crl_final.pth").write_bytes(b"final")
        resolved = resolve_source_checkpoint(run)
        assert resolved.name == "crl_best.pth"

    def test_falls_back_to_final(self, tmp_path):
        run = _write_fake_run(tmp_path, "run", "morlet_per_sensor")
        (run / "crl" / "crl_best.pth").unlink()
        (run / "crl" / "crl_final.pth").write_bytes(b"final")
        resolved = resolve_source_checkpoint(run)
        assert resolved.name == "crl_final.pth"

    def test_raises_when_no_checkpoint(self, tmp_path):
        run = tmp_path / "emptyrun" / "crl"
        run.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No CRL checkpoint"):
            resolve_source_checkpoint(run.parent)


class TestParamsEqual:
    """Tolerant float comparison for morlet_per_sensor_params round-trips."""

    def test_identical_dicts_equal(self):
        a = {"audio": {"freq_min": 20.0, "w0": 6.0}}
        ok, _ = _params_equal(a, dict(a))
        assert ok

    def test_float_tolerance(self):
        a = {"audio": {"freq_min": 20.0}}
        b = {"audio": {"freq_min": 20.0 + 1e-15}}
        ok, _ = _params_equal(a, b)
        assert ok

    def test_sensor_keys_differ(self):
        a = {"audio": {}}
        b = {"audio": {}, "seismic": {}}
        ok, why = _params_equal(a, b)
        assert not ok
        assert "sensor keys differ" in why


# ---------------------------------------------------------------------------
# CRLModel.load_from_fixed_morlet_checkpoint
# ---------------------------------------------------------------------------


class TestStateDictConversion:
    def _stage1_state(self, **overrides):
        """Build a converged stage-1 state_dict for morlet_per_sensor."""
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_z=24,
            frontend_type="morlet_per_sensor",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        model = CRLModel(cfg)
        # Simulate "trained" state by randomizing encoder weights.
        with torch.no_grad():
            for p in model.encoders.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        return cfg, model.state_dict()

    def _stage2_model(self, **overrides):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_z=24,
            frontend_type="morlet_learnable",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg, CRLModel(cfg)

    def test_log_scales_initialized_from_init_scales(self):
        _, source = self._stage1_state()
        _, model = self._stage2_model()
        model.load_from_fixed_morlet_checkpoint(source, strict=True)
        for sensor in model.sensors:
            bank = model.frontends[sensor][0]
            assert torch.allclose(
                bank.log_scales.exp(),
                bank.init_scales,
                atol=1e-5,
            ), f"scales drift from init on {sensor}"

    def test_encoder_weights_preserved(self):
        """The whole point: encoder state must round-trip."""
        _, source = self._stage1_state()
        _, model = self._stage2_model()
        # Grab one encoder weight key to check.
        enc_key = next(k for k in source if k.startswith("encoders.audio.") and "weight" in k)
        source_weight = source[enc_key].clone()
        model.load_from_fixed_morlet_checkpoint(source, strict=True)
        target_state = model.state_dict()
        assert torch.equal(target_state[enc_key], source_weight)

    def test_probe_heads_reinit(self):
        """pres_heads / type_heads / prox_heads must NOT carry over from stage 1."""
        _, source = self._stage1_state()
        # Mark a probe-head weight so we can detect leakage.
        marker_key = next(k for k in source if k.startswith("pres_heads."))
        source[marker_key] = torch.full_like(source[marker_key], 9999.0)

        _, model = self._stage2_model()
        before = model.state_dict()[marker_key].clone()
        model.load_from_fixed_morlet_checkpoint(source, strict=True)
        after = model.state_dict()[marker_key]
        # Probe heads must be UNCHANGED from their fresh init (not 9999.0).
        assert torch.equal(before, after)
        assert not torch.allclose(after, torch.tensor(9999.0))

    def test_kernel_buffers_from_source_are_dropped(self):
        """Source's kernel_re/kernel_im buffers must not appear on the
        learnable model (which has no such buffers)."""
        _, source = self._stage1_state()
        assert any("kernel_re" in k for k in source)  # precondition
        _, model = self._stage2_model()
        # Should not raise despite source having extra keys.
        model.load_from_fixed_morlet_checkpoint(source, strict=True)
        target_state = dict(model.state_dict())
        assert not any("kernel_re" in k for k in target_state)
        assert not any("kernel_im" in k for k in target_state)


# ---------------------------------------------------------------------------
# Trainer stage-2 LR groups + schedule
# ---------------------------------------------------------------------------


class TestStage2Trainer:
    def _cfg(self, **overrides):
        base = {
            "d_model": 32,
            "n_layers": 1,
            "frontend_type": "morlet_learnable",
            "d_z": 24,
            "lr": 1e-3,
            "morlet_learnable_lr_mult": 0.1,
            "stage2_encoder_lr_mult": 0.3,
        }
        base.update(overrides)
        return CRLConfig(**base)

    def test_encoder_lr_reduced_in_stage2(self, tmp_path):
        """Backbone group has `initial_lr = lr * stage2_encoder_lr_mult`;
        filter group has `initial_lr = lr * morlet_learnable_lr_mult`.
        Current `lr` may be lower if warmup is active at epoch 0."""
        cfg = self._cfg()
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path, stage2=True)
        groups = {g.get("name", ""): g for g in trainer.optimizer.param_groups}
        # initial_lr (set by LambdaLR at construction) is the peak LR before
        # any schedule multiplier is applied. That's the value config promised.
        assert groups["backbone"]["initial_lr"] == 1e-3 * 0.3
        assert groups["learnable_morlet"]["initial_lr"] == 1e-3 * 0.1
        # And at epoch 0: backbone is at peak (cosine(0) = 1.0), filter is
        # in warmup at 1/3 peak.
        assert groups["backbone"]["lr"] == 1e-3 * 0.3
        assert abs(groups["learnable_morlet"]["lr"] - 1e-3 * 0.1 * (1 / 3)) < 1e-9

    def test_stage2_false_matches_legacy_behavior(self, tmp_path):
        """When stage2=False, backbone LR is the base LR (unchanged)."""
        cfg = self._cfg()
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path, stage2=False)
        groups = {g.get("name", ""): g for g in trainer.optimizer.param_groups}
        assert groups["backbone"]["lr"] == 1e-3

    def test_warmup_ramps_filter_lr(self, tmp_path):
        """Filter LR rises linearly over the first 3 epochs (warmup), hits
        peak at epoch 3 (first post-warmup step), then cosine-anneals."""
        cfg = self._cfg(n_epochs=10)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path, stage2=True)
        filter_idx = next(
            i
            for i, g in enumerate(trainer.optimizer.param_groups)
            if g.get("name") == "learnable_morlet"
        )
        # Snapshot LR before each scheduler.step() call. After step() the
        # scheduler advances: LambdaLR applies lambda(epoch=i+1).
        filter_lrs: list[float] = []
        for _ in range(7):
            filter_lrs.append(trainer.optimizer.param_groups[filter_idx]["lr"])
            trainer.scheduler.step()

        peak_lr = 1e-3 * 0.1
        # Warmup: epochs 0,1,2 should be 1/3, 2/3, 3/3 of peak.
        assert abs(filter_lrs[0] - peak_lr / 3) < 1e-9
        assert abs(filter_lrs[1] - peak_lr * 2 / 3) < 1e-9
        assert abs(filter_lrs[2] - peak_lr) < 1e-9
        # Epoch 3 is the first cosine step (progress=0 → full peak again).
        assert abs(filter_lrs[3] - peak_lr) < 1e-9
        # Epoch 4+ must be strictly less than peak (cosine decreasing).
        assert filter_lrs[4] < peak_lr, f"filter LR not annealing post-warmup: {filter_lrs}"
        assert filter_lrs[5] < filter_lrs[4]
