import pytest
import torch
import torch.nn.functional as F
from crl_vehicle.models.intervention import label_change_target, UnknownInterventionClassifier
from crl_vehicle.models.latent import CausalLatentSpace


class TestLabelChangeTarget:

    def test_shape_and_dtype(self):
        t = label_change_target(
            torch.tensor([0, 1, 1, 0]), torch.tensor([1, 1, 0, 0]),
            torch.tensor([0, 1, 2, -1]), torch.tensor([1, 1, 2, -1]),
        )
        assert t.shape == (4, 2)
        assert t.dtype == torch.float32

    def test_pres_changed(self):
        t = label_change_target(
            torch.tensor([0, 1, 1, 0]), torch.tensor([1, 1, 0, 0]),
            torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
        )
        assert torch.allclose(t[:, 0], torch.tensor([1., 0., 1., 0.]))

    def test_type_unchanged_when_same(self):
        t = label_change_target(
            torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long),
            torch.tensor([2, 3]), torch.tensor([2, 3]),
        )
        assert t[0, 1].item() == 0.0
        assert t[1, 1].item() == 0.0

    def test_values_binary(self):
        t = label_change_target(
            torch.randint(0, 2, (16,)), torch.randint(0, 2, (16,)),
            torch.randint(-2, 4, (16,)), torch.randint(-2, 4, (16,)),
        )
        assert set(t.unique().tolist()).issubset({0.0, 1.0})


class TestUnknownInterventionClassifier:

    @pytest.fixture
    def clf(self):
        return UnknownInterventionClassifier(d_env=6, hidden_dim=64)

    def test_output_shape(self, clf):
        logits = clf(torch.randn(8, 6), torch.randn(8, 6))
        assert logits.shape == (8, 2)

    def test_input_dim_matches_d_env(self, clf):
        assert clf.classifier[0].in_features == 2 * CausalLatentSpace.D_ENV

    def test_finite(self, clf):
        logits = clf(torch.randn(8, 6), torch.randn(8, 6))
        assert logits.isfinite().all()

    def test_backprop(self, clf):
        z_t  = torch.randn(4, 6, requires_grad=True)
        z_tn = torch.randn(4, 6, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(
            clf(z_t, z_tn), torch.randint(0, 2, (4, 2)).float()
        )
        loss.backward()
        assert z_t.grad is not None
