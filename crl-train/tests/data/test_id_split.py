"""Unit tests for ID split manifest builder."""
import pytest
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.id_split import compute_split_intervals


def test_use_id_split_field_defaults_false():
    cfg = CRLConfig()
    assert cfg.use_id_split is False


def test_use_id_split_field_settable():
    cfg = CRLConfig(use_id_split=True)
    assert cfg.use_id_split is True


class TestComputeSplitIntervalsEvenSplit:
    def test_returns_val_and_test_keys(self):
        result = compute_split_intervals(n_paired=10)
        assert set(result.keys()) == {"val", "test"}

    def test_even_n_pair_splits_exactly_half(self):
        result = compute_split_intervals(n_paired=10)
        assert result["val"] == [(0, 5)]
        assert result["test"] == [(5, 10)]

    def test_odd_n_pair_floors_val(self):
        # 11 // 2 == 5 → val = [0, 5), test = [5, 11)
        result = compute_split_intervals(n_paired=11)
        assert result["val"] == [(0, 5)]
        assert result["test"] == [(5, 11)]


class TestComputeSplitIntervalsTooSmall:
    def test_n_pair_lt_2_returns_none(self):
        # Cannot split a 1-window file in half
        assert compute_split_intervals(n_paired=1) is None
        assert compute_split_intervals(n_paired=0) is None

    def test_n_pair_2_works(self):
        # Boundary: 2 windows → val=[0,1), test=[1,2)
        result = compute_split_intervals(n_paired=2)
        assert result["val"] == [(0, 1)]
        assert result["test"] == [(1, 2)]
