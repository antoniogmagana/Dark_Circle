"""Tests for ContrastiveTrainingMode — Checkpoint 3.

Coverage:
  1. Factory dispatch on training_mode='contrastive'.
  2. forward_pair produces a finite loss and gradients flow to both
     the encoder backbone and the mode's projection head.
  3. Checkpoint selection writes crl_best.pth on improvement, patience
     increments on no improvement.
  4. End-to-end smoke: Trainer.train_crl runs two epochs under
     training_mode='contrastive' and emits crl_best.pth, plus a
     train_downstream pass successfully loads that checkpoint as a
     frozen backbone (proves the whole pipeline connects).
"""

from __future__ import annotations

import pytest
import torch
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    STRATUM_CONSEC,
    STRATUM_CROSS_DS,
    STRATUM_DIFF_TYPE,
    STRATUM_SAME_TYPE,
)
from crl_vehicle.training_modes import (
    CheckpointState,
    ContrastiveTrainingMode,
    build_training_mode,
)
from training.trainer import CRLModel, Trainer


def _pair_batch(B=4, n_partners=4, audio_W=16000, seismic_W=100, strata=None):
    """Synthetic pair batch matching StratifiedPairDataset output schema."""
    if strata is None:
        strata = [
            STRATUM_CONSEC,
            STRATUM_SAME_TYPE,
            STRATUM_DIFF_TYPE,
            STRATUM_CROSS_DS,
        ]
    batch = {
        "x_audio_t": torch.randn(B, 1, audio_W) * 0.01,
        "x_seismic_t": torch.randn(B, 1, seismic_W) * 0.01,
        "audio_avail": torch.ones(B, dtype=torch.bool),
        "seismic_avail": torch.ones(B, dtype=torch.bool),
        "detection_label_t": torch.randint(0, 2, (B,)),
        "vehicle_type_t": torch.randint(0, 4, (B,)),
        "n_partners": n_partners,
    }
    for p in range(n_partners):
        batch[f"x_audio_p{p}"] = torch.randn(B, 1, audio_W) * 0.01
        batch[f"x_seismic_p{p}"] = torch.randn(B, 1, seismic_W) * 0.01
        batch[f"detection_label_p{p}"] = torch.randint(0, 2, (B,))
        batch[f"vehicle_type_p{p}"] = torch.randint(0, 4, (B,))
        batch[f"partner_stratum_p{p}"] = torch.full((B,), strata[p % len(strata)])
    return batch


@pytest.fixture
def cfg_ms():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="multiscale",
        fused_seq_len=16,
        d_z=24,
        training_mode="contrastive",
        contrastive_d_proj=32,
        contrastive_temperature=0.1,
    )


@pytest.fixture
def cfg_morlet():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet",
        d_z=24,
        training_mode="contrastive",
        contrastive_d_proj=32,
        contrastive_temperature=0.1,
    )


@pytest.fixture
def cfg_morlet_fused():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet_fused",
        fused_seq_len=16,
        d_z=24,
        training_mode="contrastive",
        contrastive_d_proj=32,
        contrastive_temperature=0.1,
    )


@pytest.fixture
def cfg_morlet_learnable():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet_learnable",
        d_z=24,
        training_mode="contrastive",
        contrastive_d_proj=32,
        contrastive_temperature=0.1,
    )


@pytest.fixture
def cfg_morlet_learnable_fused():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet_learnable_fused",
        fused_seq_len=16,
        d_z=24,
        training_mode="contrastive",
        contrastive_d_proj=32,
        contrastive_temperature=0.1,
    )


class TestFactory:
    def test_contrastive_builds(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, ContrastiveTrainingMode)

    def test_contrastive_rejects_nondefault_prior(self, cfg_ms):
        cfg_ms.prior_type = "conditional"
        with pytest.raises(ValueError, match="does not use a prior"):
            build_training_mode(cfg_ms)

    def test_contrastive_accepts_standard_prior_default(self, cfg_ms):
        # 'standard' is the default — explicit 'standard' must NOT raise.
        cfg_ms.prior_type = "standard"
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, ContrastiveTrainingMode)


class TestForwardPair:
    def test_finite_loss_multiscale(self, cfg_ms):
        model = CRLModel(cfg_ms)
        mode = ContrastiveTrainingMode(cfg_ms)
        batch = _pair_batch()
        loss, metrics = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert torch.isfinite(loss).item()
        assert loss.item() >= 0.0
        assert "contrastive_loss" in metrics

    def test_finite_loss_morlet(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        mode = ContrastiveTrainingMode(cfg_morlet)
        batch = _pair_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert torch.isfinite(loss).item()

    def test_finite_loss_morlet_fused(self, cfg_morlet_fused):
        """morlet_fused should route through _encode_fused (same as multiscale)."""
        model = CRLModel(cfg_morlet_fused)
        mode = ContrastiveTrainingMode(cfg_morlet_fused)
        batch = _pair_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert torch.isfinite(loss).item()
        assert model.is_fused_frontend() is True

    def test_finite_loss_morlet_learnable(self, cfg_morlet_learnable):
        model = CRLModel(cfg_morlet_learnable)
        mode = ContrastiveTrainingMode(cfg_morlet_learnable)
        batch = _pair_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert torch.isfinite(loss).item()
        assert model.is_fused_frontend() is False

    def test_finite_loss_morlet_learnable_fused(self, cfg_morlet_learnable_fused):
        model = CRLModel(cfg_morlet_learnable_fused)
        mode = ContrastiveTrainingMode(cfg_morlet_learnable_fused)
        batch = _pair_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert torch.isfinite(loss).item()
        assert model.is_fused_frontend() is True

    def test_gradient_reaches_encoder_and_projection(self, cfg_ms):
        model = CRLModel(cfg_ms)
        mode = ContrastiveTrainingMode(cfg_ms)
        batch = _pair_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        loss.backward()

        enc_grad_seen = any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.encoder.parameters()
        )
        proj_grad_seen = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in mode.projection.parameters()
        )
        assert enc_grad_seen, "backbone encoder received no gradient"
        assert proj_grad_seen, "projection head received no gradient"

    def test_empty_batch_returns_zero(self, cfg_ms):
        model = CRLModel(cfg_ms)
        mode = ContrastiveTrainingMode(cfg_ms)
        batch = _pair_batch()
        batch["audio_avail"] = torch.zeros(batch["x_audio_t"].shape[0], dtype=torch.bool)
        batch["seismic_avail"] = torch.zeros(batch["x_audio_t"].shape[0], dtype=torch.bool)
        loss, _ = mode.forward_pair(model, batch, beta=0.0, device=torch.device("cpu"))
        assert loss.item() == 0.0


class TestCheckpointSelection:
    def test_saves_on_improvement(self, cfg_ms):
        mode = ContrastiveTrainingMode(cfg_ms)
        state = CheckpointState()
        saves = mode.should_save_checkpoint(
            {"val_contrastive_loss": 5.0},
            epoch=0,
            state=state,
        )
        assert saves == {"crl_best.pth": True}
        assert state.bests["val_contrastive_loss"] == 5.0
        assert state.patience_count == 0

    def test_skips_on_no_improvement(self, cfg_ms):
        mode = ContrastiveTrainingMode(cfg_ms)
        state = CheckpointState()
        mode.should_save_checkpoint({"val_contrastive_loss": 5.0}, 0, state)
        saves = mode.should_save_checkpoint(
            {"val_contrastive_loss": 5.5},
            epoch=1,
            state=state,
        )
        assert saves == {"crl_best.pth": False}
        assert state.patience_count == 1

    def test_early_stop_config(self, cfg_ms):
        mode = ContrastiveTrainingMode(cfg_ms)
        assert mode.early_stop_metric() == "val_contrastive_loss"
        assert mode.early_stop_mode() == "min"


class _PairLoader:
    """Minimal iterable yielding fixed synthetic pair batches."""

    def __init__(self, n_batches=2, strata=None):
        self.n = n_batches
        self.strata = strata

    def __iter__(self):
        for _ in range(self.n):
            yield _pair_batch(B=4, n_partners=4, strata=self.strata)


class _SingleLoader:
    """Minimal iterable yielding a single-sample batch for train_downstream."""

    def __init__(self, n_batches=2):
        self.n = n_batches

    def __iter__(self):
        B = 4
        for _ in range(self.n):
            yield {
                "x_audio": torch.randn(B, 1, 16000) * 0.01,
                "x_seismic": torch.randn(B, 1, 100) * 0.01,
                "audio_avail": torch.ones(B, dtype=torch.bool),
                "seismic_avail": torch.ones(B, dtype=torch.bool),
                "detection_label": torch.randint(0, 2, (B,)),
                "vehicle_type": torch.randint(0, 4, (B,)),
                "segment_id": torch.zeros(B, dtype=torch.long),
            }


class TestEndToEnd:
    def test_train_crl_then_downstream(self, cfg_ms, tmp_path):
        """Smoke: contrastive CRL → crl_best.pth → train_downstream loads it."""
        model = CRLModel(cfg_ms)
        trainer = Trainer(model, cfg_ms, torch.device("cpu"), tmp_path)

        trainer.train_crl(
            train_loader=_PairLoader(n_batches=2),
            val_loader=_PairLoader(n_batches=1),
            epochs=2,
        )

        ckpt = tmp_path / "crl_best.pth"
        assert ckpt.exists(), "contrastive CRL did not produce crl_best.pth"
        assert (tmp_path / "crl_final.pth").exists()
        assert (tmp_path / "crl_metrics.csv").exists()

        # Now exercise the downstream-load path. `train_downstream` reads
        # the checkpoint file named by `ckpt_name` from its save_dir.
        trainer.train_downstream(
            train_loader=_SingleLoader(n_batches=2),
            val_loader=_SingleLoader(n_batches=1),
            epochs=1,
            ckpt_name="crl_best.pth",
        )
        assert (tmp_path / "downstream_metrics.csv").exists()
