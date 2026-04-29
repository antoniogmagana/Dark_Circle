"""Tests for DisentangledVAETrainingMode.

Coverage:
  1. Factory dispatch on training_mode='disentangled'.
  2. Factory rejects non-standard prior with disentangled mode.
  3. forward_pair returns finite loss + populated metric dict for both
     fused (multiscale) and per-sensor (morlet_per_sensor) frontends.
  4. Gradients flow to encoder backbone AND mode-owned heads.
  5. Cross-modal alignment is non-zero for per-sensor when both modalities
     present, and skipped (0) for fused frontends.
  6. Checkpoint selection: dual ckpt by val_ref_elbo + val_aux_type_f1.
  7. update_beta returns adaptive-beta tuple shape.
"""
from __future__ import annotations

import torch
import pytest

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    STRATUM_CONSEC, STRATUM_CROSS_DS, STRATUM_DIFF_TYPE, STRATUM_SAME_TYPE,
)
from crl_vehicle.training_modes import (
    CheckpointState, DisentangledVAETrainingMode, build_training_mode,
)
from training.trainer import CRLModel


def _pair_batch(B=4, n_partners=2, audio_W=16000, seismic_W=200,
                strata=None):
    """Synthetic pair batch matching StratifiedPairDataset output schema."""
    if strata is None:
        strata = [STRATUM_CONSEC, STRATUM_SAME_TYPE]
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
        batch[f"partner_stratum_p{p}"] = torch.full((B,), strata[p % len(strata)])
    return batch


@pytest.fixture
def cfg_ms():
    return CRLConfig(
        d_model=32, n_layers=1, n_heads=4,
        frontend_type="multiscale", fused_seq_len=16, d_z=24,
        training_mode="disentangled", d_signal=12,
        lambda_align=1.0, lambda_stab=0.1, lambda_interv_inv=1.0,
    )


@pytest.fixture
def cfg_per_sensor():
    return CRLConfig(
        d_model=32, n_layers=1, n_heads=4,
        frontend_type="morlet_per_sensor", fused_seq_len=16, d_z=24,
        training_mode="disentangled", d_signal=12,
        lambda_align=1.0, lambda_stab=0.1, lambda_interv_inv=1.0,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_dispatches_to_disentangled(cfg_ms):
    mode = build_training_mode(cfg_ms)
    assert isinstance(mode, DisentangledVAETrainingMode)
    assert mode.latent.d_signal == 12
    assert mode.latent.d_env == 12


def test_factory_rejects_non_standard_prior():
    cfg = CRLConfig(training_mode="disentangled", prior_type="conditional")
    with pytest.raises(ValueError, match="standard"):
        build_training_mode(cfg)


def test_pres_and_type_heads_sized_to_d_signal(cfg_ms):
    mode = build_training_mode(cfg_ms)
    # Pres head: Linear(d_signal, 1)
    pres_w = mode.pres_head.head.weight
    assert pres_w.shape == (1, cfg_ms.d_signal)
    # Type head: Linear(d_signal, 4)
    type_w = mode.type_head.head.weight
    assert type_w.shape == (4, cfg_ms.d_signal)


# ---------------------------------------------------------------------------
# forward_pair — fused (multiscale)
# ---------------------------------------------------------------------------

def test_forward_pair_fused_returns_finite_loss(cfg_ms):
    mode = build_training_mode(cfg_ms)
    model = CRLModel(config=cfg_ms)
    batch = _pair_batch()
    loss, metrics = mode.forward_pair(model, batch, beta=0.5, device=torch.device("cpu"))
    assert torch.isfinite(loss)
    assert loss.requires_grad
    for k in ("recon", "kl", "raw_kl", "align", "stab", "interv_inv", "total"):
        assert k in metrics
    # Fused: alignment is n/a → must be 0.
    assert metrics["align"] == 0.0


def test_forward_pair_fused_grad_flows_to_backbone_and_heads(cfg_ms):
    mode = build_training_mode(cfg_ms)
    model = CRLModel(config=cfg_ms)
    batch = _pair_batch()
    loss, _ = mode.forward_pair(model, batch, beta=0.5, device=torch.device("cpu"))
    loss.backward()

    # Backbone params should have grads.
    backbone_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in backbone_grads)

    # Mode head params should have grads.
    pres_grad = mode.pres_head.head.weight.grad
    type_grad = mode.type_head.head.weight.grad
    assert pres_grad is not None and pres_grad.abs().sum() > 0
    assert type_grad is not None and type_grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# forward_pair — per-sensor (morlet_per_sensor)
# ---------------------------------------------------------------------------

def test_forward_pair_per_sensor_returns_finite_loss(cfg_per_sensor):
    mode = build_training_mode(cfg_per_sensor)
    model = CRLModel(config=cfg_per_sensor)
    batch = _pair_batch()
    loss, metrics = mode.forward_pair(model, batch, beta=0.5, device=torch.device("cpu"))
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_forward_pair_per_sensor_alignment_nonzero_when_both_modalities(cfg_per_sensor):
    mode = build_training_mode(cfg_per_sensor)
    model = CRLModel(config=cfg_per_sensor)
    batch = _pair_batch()  # both modalities available everywhere
    _, metrics = mode.forward_pair(model, batch, beta=0.5, device=torch.device("cpu"))
    # With both modalities and random init, alignment should be > 0.
    assert metrics["align"] > 0.0


def test_forward_pair_per_sensor_grad_flows(cfg_per_sensor):
    mode = build_training_mode(cfg_per_sensor)
    model = CRLModel(config=cfg_per_sensor)
    batch = _pair_batch()
    loss, _ = mode.forward_pair(model, batch, beta=0.5, device=torch.device("cpu"))
    loss.backward()
    # Mode heads have grads.
    assert mode.pres_head.head.weight.grad is not None
    assert mode.pres_head.head.weight.grad.abs().sum() > 0
    assert mode.type_head.head.weight.grad is not None
    assert mode.type_head.head.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def test_should_save_checkpoint_dual_metric():
    cfg = CRLConfig(training_mode="disentangled", d_z=24, d_signal=12)
    mode = build_training_mode(cfg)
    state = CheckpointState()

    # First epoch: both improve from defaults, both saved.
    saves = mode.should_save_checkpoint(
        {"val_ref_elbo": 5.0, "val_aux_type_f1": 0.4}, epoch=0, state=state
    )
    assert saves[mode.CKPT_REF_ELBO] is True
    assert saves[mode.CKPT_AUX_TYPE_F1] is True
    assert state.bests["val_ref_elbo"] == 5.0
    assert state.bests["val_aux_type_f1"] == 0.4

    # Second epoch: ref_elbo worse, aux better → only aux saved.
    saves = mode.should_save_checkpoint(
        {"val_ref_elbo": 6.0, "val_aux_type_f1": 0.45}, epoch=1, state=state
    )
    assert saves[mode.CKPT_REF_ELBO] is False
    assert saves[mode.CKPT_AUX_TYPE_F1] is True
    assert state.patience_count == 1

    # Third epoch: ref_elbo better, aux worse → only ref saved, patience reset.
    saves = mode.should_save_checkpoint(
        {"val_ref_elbo": 4.0, "val_aux_type_f1": 0.4}, epoch=2, state=state
    )
    assert saves[mode.CKPT_REF_ELBO] is True
    assert saves[mode.CKPT_AUX_TYPE_F1] is False
    assert state.patience_count == 0


def test_early_stop_metric():
    cfg = CRLConfig(training_mode="disentangled")
    mode = build_training_mode(cfg)
    assert mode.early_stop_metric() == "val_ref_elbo"
    assert mode.early_stop_mode() == "min"


# ---------------------------------------------------------------------------
# update_beta — adaptive schedule on full-z KL
# ---------------------------------------------------------------------------

def test_update_beta_increases_when_kl_above_target():
    cfg = CRLConfig(training_mode="disentangled",
                    beta_step=0.02, kl_floor=0.01, kl_target=0.5)
    mode = build_training_mode(cfg)
    state = CheckpointState()
    state.prev_val_recon = 10.0
    new_beta, event = mode.update_beta(
        beta=0.0, val_m={"val_raw_kl": 1.0, "val_recon": 5.0},
        state=state, config=cfg,
    )
    assert new_beta == pytest.approx(0.02)
    assert event == "↑"


def test_update_beta_collapses_when_kl_below_floor():
    cfg = CRLConfig(training_mode="disentangled",
                    beta_step=0.02, kl_floor=0.01, kl_target=0.5)
    mode = build_training_mode(cfg)
    state = CheckpointState()
    state.prev_val_recon = 10.0
    new_beta, event = mode.update_beta(
        beta=0.5, val_m={"val_raw_kl": 0.005, "val_recon": 5.0},
        state=state, config=cfg,
    )
    assert new_beta == pytest.approx(0.48)
    assert event == "↓collapse"


# ---------------------------------------------------------------------------
# val_metrics_summary derives ref_elbo
# ---------------------------------------------------------------------------

def test_val_metrics_summary_computes_ref_elbo():
    cfg = CRLConfig(training_mode="disentangled")
    mode = build_training_mode(cfg)
    val_m = {"val_recon": 3.0, "val_raw_kl": 12.0}
    out = mode.val_metrics_summary(val_m)
    assert out["val_ref_elbo"] == 15.0
