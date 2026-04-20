import pytest
import torch
from crl_vehicle.models.latent import CausalLatentSpace


@pytest.fixture
def latent():
    return CausalLatentSpace(d_z=24)


def test_class_constants():
    assert CausalLatentSpace.D_PRES == 4
    assert CausalLatentSpace.D_TYPE == 6
    assert CausalLatentSpace.D_PROX == 3
    assert CausalLatentSpace.D_ENV  == 6
    assert CausalLatentSpace.D_CAUSAL == 19


def test_default_d_z_gives_d_free_5(latent):
    assert latent.d_z == 24
    assert latent.d_free == 5


def test_split_shapes(latent):
    z = torch.zeros(8, 24)
    z_pres, z_type, z_prox, z_env, z_free = latent.split(z)
    assert z_pres.shape == (8, 4)
    assert z_type.shape == (8, 6)
    assert z_prox.shape == (8, 3)
    assert z_env.shape  == (8, 6)
    assert z_free.shape == (8, 5)


def test_split_contiguous_partition(latent):
    z = torch.arange(24, dtype=torch.float).unsqueeze(0)
    z_pres, z_type, z_prox, z_env, z_free = latent.split(z)
    assert z_pres[0, -1].item() == 3.0   # dims 0-3
    assert z_type[0,  0].item() == 4.0   # dims 4-9
    assert z_prox[0,  0].item() == 10.0  # dims 10-12
    assert z_env[0,   0].item() == 13.0  # dims 13-18
    assert z_free[0,  0].item() == 19.0  # dims 19-23


def test_split_3d_input(latent):
    z = torch.zeros(4, 3, 24)
    z_pres, *_ = latent.split(z)
    assert z_pres.shape == (4, 3, 4)


def test_larger_d_z_expands_only_free_subspace():
    latent = CausalLatentSpace(d_z=32)
    assert latent.d_z == 32
    assert latent.d_free == 13

    z = torch.zeros(2, 32)
    z_pres, z_type, z_prox, z_env, z_free = latent.split(z)
    assert z_pres.shape == (2, 4)
    assert z_type.shape == (2, 6)
    assert z_prox.shape == (2, 3)
    assert z_env.shape  == (2, 6)
    assert z_free.shape == (2, 13)


def test_d_z_at_or_below_causal_budget_raises():
    with pytest.raises(ValueError, match="d_z > 19"):
        CausalLatentSpace(d_z=19)
    with pytest.raises(ValueError, match="d_z > 19"):
        CausalLatentSpace(d_z=16)


def test_no_trainable_parameters(latent):
    assert sum(p.numel() for p in latent.parameters()) == 0
