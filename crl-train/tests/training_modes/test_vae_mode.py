"""Tests for VAETrainingMode — the Checkpoint 1 reference implementation.

Validation strategy:
  1. Interface — the mode implements everything the Trainer needs.
  2. Forward parity — per-batch loss & metrics match what the old Trainer
     produced on the same input. We can't bit-compare across the refactor,
     but we can verify the metric schema matches and values are finite.
  3. Checkpoint selection — dual-checkpoint logic matches the old behavior.
  4. Factory — build_training_mode returns the right instance.
"""
from __future__ import annotations

import torch
import pytest

from crl_vehicle.config import CRLConfig
from crl_vehicle.priors import StandardPrior
from crl_vehicle.training_modes import (
    CheckpointState, TrainingMode, VAETrainingMode, build_training_mode,
)
from training.trainer import CRLModel


def _synthetic_batch(B=4, n_partners=4, audio_W=16000, seismic_W=200):
    batch = {
        "x_audio_t":         torch.randn(B, 1, audio_W) * 0.01,
        "x_seismic_t":       torch.randn(B, 1, seismic_W) * 0.01,
        "audio_avail":       torch.ones(B, dtype=torch.bool),
        "seismic_avail":     torch.ones(B, dtype=torch.bool),
        "detection_label_t": torch.randint(0, 2, (B,)),
        "vehicle_type_t":    torch.randint(0, 4, (B,)),
        "n_partners":        n_partners,
    }
    for p in range(n_partners):
        batch[f"x_audio_p{p}"]         = torch.randn(B, 1, audio_W) * 0.01
        batch[f"x_seismic_p{p}"]       = torch.randn(B, 1, seismic_W) * 0.01
        batch[f"detection_label_p{p}"] = torch.randint(0, 2, (B,))
        batch[f"vehicle_type_p{p}"]    = torch.randint(0, 4, (B,))
        batch[f"partner_stratum_p{p}"] = torch.full((B,), p % 4)
    return batch


@pytest.fixture
def cfg_ms():
    return CRLConfig(d_model=32, n_layers=1, n_heads=4,
                     frontend_type="multiscale", fused_seq_len=16, d_z=24)


@pytest.fixture
def cfg_morlet():
    return CRLConfig(d_model=32, n_layers=1, n_heads=4,
                     frontend_type="morlet", d_z=24)


class TestFactory:
    def test_default_config_builds_vae_standard(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, VAETrainingMode)
        assert isinstance(mode.prior, StandardPrior)

    def test_unknown_training_mode_raises(self, cfg_ms):
        cfg_ms.training_mode = "nope"
        with pytest.raises(ValueError, match="Unknown training_mode"):
            build_training_mode(cfg_ms)

    def test_unknown_prior_type_raises(self, cfg_ms):
        cfg_ms.prior_type = "nope"
        with pytest.raises(ValueError, match="Unknown prior_type"):
            build_training_mode(cfg_ms)

    def test_conditional_prior_not_yet_implemented(self, cfg_ms):
        """Checkpoint 2 adds this; Checkpoint 1 must refuse cleanly."""
        cfg_ms.prior_type = "conditional"
        with pytest.raises(ValueError, match="Checkpoint 2"):
            build_training_mode(cfg_ms)


class TestVAEModeInterface:
    def test_is_training_mode(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, TrainingMode)

    def test_early_stop_metric_is_ref_elbo(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert mode.early_stop_metric() == "val_ref_elbo"
        assert mode.early_stop_mode() == "min"


class TestForwardPairMetrics:
    """forward_pair must produce the scalar + tensor keys Trainer consumes."""

    EXPECTED_SCALAR_KEYS = {"recon", "kl", "raw_kl", "interv", "total"}
    EXPECTED_TENSOR_KEYS = {
        "aux_pres_logits", "aux_pres_labels",
        "aux_type_logits", "aux_type_labels",
    }

    @pytest.mark.parametrize("cfg_name", ["cfg_ms", "cfg_morlet"])
    def test_returns_loss_and_metrics(self, cfg_name, request):
        cfg = request.getfixturevalue(cfg_name)
        mode = build_training_mode(cfg)
        model = CRLModel(cfg)
        batch = _synthetic_batch()
        loss, metrics = mode.forward_pair(model, batch, beta=0.1,
                                          device=torch.device("cpu"))
        assert torch.isfinite(loss)
        assert self.EXPECTED_SCALAR_KEYS.issubset(metrics.keys())
        assert self.EXPECTED_TENSOR_KEYS.issubset(metrics.keys())

    @pytest.mark.parametrize("cfg_name", ["cfg_ms", "cfg_morlet"])
    def test_gradients_flow(self, cfg_name, request):
        cfg = request.getfixturevalue(cfg_name)
        mode = build_training_mode(cfg)
        model = CRLModel(cfg)
        batch = _synthetic_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.1,
                                    device=torch.device("cpu"))
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not p.grad.isfinite().all()]
        assert not bad, f"Non-finite grads: {bad}"


class TestCheckpointLogic:
    def test_first_epoch_saves_both(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState()
        val_m = {"val_ref_elbo": 5.0, "val_aux_type_f1": 0.5,
                 "val_recon": 0.1, "val_raw_kl": 4.9}
        saves = mode.should_save_checkpoint(val_m, epoch=0, state=state)
        assert saves[mode.CKPT_REF_ELBO] is True
        assert saves[mode.CKPT_AUX_TYPE_F1] is True
        assert state.bests["val_ref_elbo"] == 5.0
        assert state.bests["val_aux_type_f1"] == 0.5

    def test_ref_elbo_saves_when_improves(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0,
                                        "val_aux_type_f1": 0.5})
        saves = mode.should_save_checkpoint(
            {"val_ref_elbo": 4.5, "val_aux_type_f1": 0.4,
             "val_recon": 0.1, "val_raw_kl": 4.4},
            epoch=1, state=state,
        )
        assert saves[mode.CKPT_REF_ELBO] is True
        assert saves[mode.CKPT_AUX_TYPE_F1] is False

    def test_aux_type_saves_when_improves(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0,
                                        "val_aux_type_f1": 0.5})
        saves = mode.should_save_checkpoint(
            {"val_ref_elbo": 6.0, "val_aux_type_f1": 0.7,
             "val_recon": 0.1, "val_raw_kl": 5.9},
            epoch=1, state=state,
        )
        assert saves[mode.CKPT_REF_ELBO] is False
        assert saves[mode.CKPT_AUX_TYPE_F1] is True

    def test_patience_increments_on_no_ref_elbo_improvement(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0,
                                        "val_aux_type_f1": 0.5})
        mode.should_save_checkpoint(
            {"val_ref_elbo": 5.1, "val_aux_type_f1": 0.4,
             "val_recon": 0.1, "val_raw_kl": 5.0},
            epoch=1, state=state,
        )
        assert state.patience_count == 1

    def test_checkpoint_summary_schema(self, cfg_ms):
        """Must match the legacy crl_checkpoint_summary.json schema."""
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(
            bests={"val_ref_elbo": 0.241628, "val_aux_type_f1": 0.7009},
            best_epochs={"val_aux_type_f1": 28},
        )
        summary = mode.checkpoint_summary(state)
        assert "best_ref_elbo" in summary
        assert "best_aux_type_f1" in summary
        assert "best_aux_type_epoch" in summary
        assert "checkpoints" in summary
        assert "crl_best.pth" in summary["checkpoints"]
        assert "crl_best_aux_type.pth" in summary["checkpoints"]
        assert "crl_final.pth" in summary["checkpoints"]
        assert summary["best_aux_type_epoch"] == 28


class TestValMetricsSummary:
    def test_computes_ref_elbo(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        out = mode.val_metrics_summary({"val_recon": 0.2, "val_raw_kl": 0.8})
        assert out["val_ref_elbo"] == pytest.approx(1.0)

    def test_preserves_other_keys(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        out = mode.val_metrics_summary({
            "val_recon": 0.2, "val_raw_kl": 0.8, "val_aux_type_f1": 0.5,
        })
        assert out["val_aux_type_f1"] == 0.5
