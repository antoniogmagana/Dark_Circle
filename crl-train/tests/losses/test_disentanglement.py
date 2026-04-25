import pytest
import torch

from crl_vehicle.losses.disentanglement import (
    cross_modal_alignment_loss,
    intervention_invariance_loss,
    temporal_stability_loss,
)


# ---------------------------------------------------------------------------
# cross_modal_alignment_loss
# ---------------------------------------------------------------------------

def test_alignment_zero_when_identical():
    mu = torch.randn(8, 12)
    loss = cross_modal_alignment_loss(mu, mu)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_alignment_positive_when_different():
    mu_a = torch.randn(8, 12)
    mu_s = torch.randn(8, 12)
    assert cross_modal_alignment_loss(mu_a, mu_s).item() > 0


def test_alignment_grad_flows_to_both():
    mu_a = torch.randn(4, 12, requires_grad=True)
    mu_s = torch.randn(4, 12, requires_grad=True)
    cross_modal_alignment_loss(mu_a, mu_s).backward()
    assert mu_a.grad is not None and mu_a.grad.abs().sum() > 0
    assert mu_s.grad is not None and mu_s.grad.abs().sum() > 0


def test_alignment_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_modal_alignment_loss(torch.zeros(4, 12), torch.zeros(4, 8))


def test_alignment_finite_on_large_inputs():
    mu_a = torch.randn(64, 32) * 100
    mu_s = torch.randn(64, 32) * 100
    loss = cross_modal_alignment_loss(mu_a, mu_s)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# temporal_stability_loss
# ---------------------------------------------------------------------------

def test_stability_zero_when_identical():
    mu = torch.randn(8, 12)
    mask = torch.ones(8, dtype=torch.bool)
    loss = temporal_stability_loss(mu, mu, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_stability_zero_when_no_valid_rows():
    mu_t  = torch.randn(8, 12, requires_grad=True)
    mu_tn = torch.randn(8, 12)
    mask = torch.zeros(8, dtype=torch.bool)
    loss = temporal_stability_loss(mu_t, mu_tn, mask)
    assert loss.item() == 0.0
    # zero loss with requires_grad still allows backward without error
    loss.backward()


def test_stability_only_counts_valid_rows():
    mu_t  = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    mu_tn = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0,  0.0]])
    # Only row 0 valid → diff is zero
    mask_zero = torch.tensor([True, False, False])
    assert temporal_stability_loss(mu_t, mu_tn, mask_zero).item() == 0.0
    # Only row 2 valid → diff is 200
    mask_nonzero = torch.tensor([False, False, True])
    assert temporal_stability_loss(mu_t, mu_tn, mask_nonzero).item() == pytest.approx(200.0)


def test_stability_grad_flows_to_both():
    mu_t  = torch.randn(4, 12, requires_grad=True)
    mu_tn = torch.randn(4, 12, requires_grad=True)
    mask = torch.ones(4, dtype=torch.bool)
    temporal_stability_loss(mu_t, mu_tn, mask).backward()
    assert mu_t.grad is not None and mu_t.grad.abs().sum() > 0
    assert mu_tn.grad is not None and mu_tn.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# intervention_invariance_loss
# ---------------------------------------------------------------------------

def test_invariance_zero_when_identical():
    mu = torch.randn(8, 12)
    loss = intervention_invariance_loss(mu, mu)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_invariance_positive_when_different():
    mu_clean = torch.randn(8, 12)
    mu_int   = torch.randn(8, 12)
    assert intervention_invariance_loss(mu_clean, mu_int).item() > 0


def test_invariance_grad_flows_to_both():
    mu_c = torch.randn(4, 12, requires_grad=True)
    mu_i = torch.randn(4, 12, requires_grad=True)
    intervention_invariance_loss(mu_c, mu_i).backward()
    assert mu_c.grad is not None and mu_c.grad.abs().sum() > 0
    assert mu_i.grad is not None and mu_i.grad.abs().sum() > 0


def test_invariance_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        intervention_invariance_loss(torch.zeros(4, 12), torch.zeros(4, 8))
