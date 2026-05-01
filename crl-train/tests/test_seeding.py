"""Tests for crl_vehicle.seeding — deterministic RNG init for reproducible runs."""

import os

import numpy as np
import torch
from crl_vehicle.seeding import seed_everything, seeded_dataloader_kwargs


class TestSeedEverything:
    def test_python_hash_seed_set(self):
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_torch_rng_deterministic(self):
        seed_everything(42)
        a = torch.randn(8)
        seed_everything(42)
        b = torch.randn(8)
        assert torch.equal(a, b)

    def test_numpy_rng_deterministic(self):
        seed_everything(42)
        a = np.random.randn(8)
        seed_everything(42)
        b = np.random.randn(8)
        assert np.array_equal(a, b)

    def test_python_random_deterministic(self):
        import random

        seed_everything(42)
        a = [random.random() for _ in range(4)]
        seed_everything(42)
        b = [random.random() for _ in range(4)]
        assert a == b

    def test_different_seeds_produce_different_streams(self):
        seed_everything(1)
        a = torch.randn(8)
        seed_everything(2)
        b = torch.randn(8)
        assert not torch.equal(a, b)


class TestSeededDataLoaderKwargs:
    def test_returns_generator_and_worker_init_fn(self):
        kwargs = seeded_dataloader_kwargs(7)
        assert isinstance(kwargs["generator"], torch.Generator)
        assert callable(kwargs["worker_init_fn"])

    def test_generator_is_seeded(self):
        # Two calls with the same seed produce generators in identical state.
        g1 = seeded_dataloader_kwargs(7)["generator"]
        g2 = seeded_dataloader_kwargs(7)["generator"]
        s1 = torch.randint(0, 1_000_000, (4,), generator=g1)
        s2 = torch.randint(0, 1_000_000, (4,), generator=g2)
        assert torch.equal(s1, s2)

    def test_different_seeds_yield_different_generator_streams(self):
        g1 = seeded_dataloader_kwargs(1)["generator"]
        g2 = seeded_dataloader_kwargs(2)["generator"]
        s1 = torch.randint(0, 1_000_000, (4,), generator=g1)
        s2 = torch.randint(0, 1_000_000, (4,), generator=g2)
        assert not torch.equal(s1, s2)

    def test_worker_init_fn_seeds_each_worker_distinctly(self):
        init = seeded_dataloader_kwargs(100)["worker_init_fn"]
        init(0)
        a = torch.randn(4)
        init(1)
        b = torch.randn(4)
        assert not torch.equal(a, b)

    def test_worker_init_fn_is_deterministic_per_worker(self):
        init1 = seeded_dataloader_kwargs(100)["worker_init_fn"]
        init1(3)
        a = torch.randn(4)
        init2 = seeded_dataloader_kwargs(100)["worker_init_fn"]
        init2(3)
        b = torch.randn(4)
        assert torch.equal(a, b)


class TestEndToEndDeterminism:
    """Two seeded runs of the same model init + forward must produce identical outputs."""

    def test_model_init_identical_under_same_seed(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, n_heads=4, frontend_type="multiscale", fused_seq_len=16, d_z=24
        )

        seed_everything(42)
        m1 = CRLModel(cfg)
        seed_everything(42)
        m2 = CRLModel(cfg)

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=False):
            assert n1 == n2
            assert torch.equal(p1, p2), f"param {n1} diverges under same seed"

    def test_model_init_differs_under_different_seed(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, n_heads=4, frontend_type="multiscale", fused_seq_len=16, d_z=24
        )

        seed_everything(1)
        m1 = CRLModel(cfg)
        seed_everything(2)
        m2 = CRLModel(cfg)

        any_diff = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=False)
        )
        assert any_diff, "different seeds produced identical models"
