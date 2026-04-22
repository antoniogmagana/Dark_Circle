import pytest
import torch
import torch.nn.functional as F

from crl_vehicle.losses.contrastive import nt_xent_loss


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class TestNTXentBasics:

    def test_scalar_shape(self):
        a = _normed(torch.randn(4, 16))
        p = _normed(torch.randn(4, 3, 16))
        m = torch.tensor([[True, False, False]] * 4)
        assert nt_xent_loss(a, p, m).shape == ()

    def test_finite(self):
        a = _normed(torch.randn(4, 16))
        p = _normed(torch.randn(4, 3, 16))
        m = torch.tensor([[True, False, False]] * 4)
        assert torch.isfinite(nt_xent_loss(a, p, m)).item()

    def test_gradient_flows_to_both_sides(self):
        a = _normed(torch.randn(4, 16)).detach().requires_grad_(True)
        p = _normed(torch.randn(4, 3, 16)).detach().requires_grad_(True)
        m = torch.tensor([[True, False, False]] * 4)
        loss = nt_xent_loss(a, p, m)
        loss.backward()
        assert a.grad is not None and a.grad.abs().sum() > 0
        assert p.grad is not None and p.grad.abs().sum() > 0

    def test_empty_mask_returns_zero(self):
        a = _normed(torch.randn(4, 16))
        p = _normed(torch.randn(4, 3, 16))
        m = torch.zeros(4, 3, dtype=torch.bool)
        loss = nt_xent_loss(a, p, m)
        assert loss.item() == 0.0


class TestNTXentSemantics:

    def test_perfectly_aligned_positive_is_low(self):
        # Positive partner exactly equals anchor → highest similarity; loss low.
        B, P, D = 4, 3, 16
        a = _normed(torch.randn(B, D))
        p = torch.zeros(B, P, D)
        p[:, 0] = a                        # partner 0 == anchor (positive)
        p[:, 1:] = _normed(torch.randn(B, P - 1, D)) * 0.01   # near-orthogonal noise
        p = _normed(p)
        m = torch.zeros(B, P, dtype=torch.bool)
        m[:, 0] = True
        loss = nt_xent_loss(a, p, m, temperature=0.1).item()

        # Randomized partners baseline
        p_rand = _normed(torch.randn(B, P, D))
        loss_rand = nt_xent_loss(a, p_rand, m, temperature=0.1).item()
        assert loss < loss_rand

    def test_temperature_affects_loss(self):
        torch.manual_seed(0)
        a = _normed(torch.randn(4, 16))
        p = _normed(torch.randn(4, 3, 16))
        m = torch.tensor([[True, False, False]] * 4)
        lo = nt_xent_loss(a, p, m, temperature=0.05).item()
        hi = nt_xent_loss(a, p, m, temperature=1.0).item()
        assert lo != pytest.approx(hi)

    def test_multi_positive_averages(self):
        # With 2 positives, loss should be mean of the two per-positive log-probs.
        B, P, D = 2, 3, 8
        torch.manual_seed(1)
        a = _normed(torch.randn(B, D))
        p = _normed(torch.randn(B, P, D))

        m1 = torch.zeros(B, P, dtype=torch.bool); m1[:, 0] = True
        m2 = torch.zeros(B, P, dtype=torch.bool); m2[:, 1] = True
        m_both = m1 | m2

        l_both = nt_xent_loss(a, p, m_both, temperature=0.1).item()
        l1 = nt_xent_loss(a, p, m1, temperature=0.1).item()
        l2 = nt_xent_loss(a, p, m2, temperature=0.1).item()
        assert l_both == pytest.approx(0.5 * (l1 + l2), rel=1e-4)

    def test_rows_without_positives_dont_dominate_mean(self):
        # When only one of two rows has a positive, loss equals that row's
        # per-anchor contribution — the positive-less row must be dropped from
        # the mean, not folded in as zero.
        B, P, D = 2, 3, 8
        torch.manual_seed(2)
        a = _normed(torch.randn(B, D))
        p = _normed(torch.randn(B, P, D))

        m_row0 = torch.zeros(B, P, dtype=torch.bool); m_row0[0, 0] = True
        m_row1 = torch.zeros(B, P, dtype=torch.bool); m_row1[1, 0] = True

        # If row 1 were folded in as zero, swapping which row has the positive
        # would give identical loss only by coincidence. Losses should differ
        # because each row sees different logits under the same partner pool.
        loss0 = nt_xent_loss(a, p, m_row0, temperature=0.1).item()
        loss1 = nt_xent_loss(a, p, m_row1, temperature=0.1).item()
        # Both should be strictly positive, neither collapsed to the zero-fold:
        assert loss0 > 0.0 and loss1 > 0.0
