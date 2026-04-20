import pytest
import torch
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead, LinearProximityHead
from crl_vehicle.models.latent import CausalLatentSpace


def test_presence_head_shape():
    head = LinearPresenceHead(d_in=CausalLatentSpace.D_PRES)
    assert head(torch.randn(8, 4)).shape == (8, 1)


def test_type_head_shape():
    assert LinearTypeHead(d_in=6, n_classes=4)(torch.randn(8, 6)).shape == (8, 4)


def test_type_head_default_d_in():
    assert LinearTypeHead().head.in_features == CausalLatentSpace.D_TYPE


def test_proximity_head_shape():
    assert LinearProximityHead(d_in=CausalLatentSpace.D_PROX)(torch.randn(8, 3)).shape == (8, 1)


def test_all_heads_finite():
    for head, z in [
        (LinearPresenceHead(), torch.randn(4, 4)),
        (LinearTypeHead(), torch.randn(4, 6)),
        (LinearProximityHead(), torch.randn(4, 3)),
    ]:
        assert head(z).isfinite().all()


def test_outputs_are_float32():
    head = LinearPresenceHead()
    assert head(torch.zeros(4, 4)).dtype == torch.float32
