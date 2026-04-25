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


from pathlib import Path
import numpy as np
import pandas as pd
from crl_vehicle.data.id_split import extract_runs


def _write_split_runs_parquet(
    path: Path, scene_run_lengths: list[tuple[int, int, int]],
) -> None:
    """Write a parquet with scene_id/run_id/amplitude columns.

    scene_run_lengths: list of (scene_id, run_id, n_samples) — written
    in order so blocks are contiguous (the normal case).
    """
    rows = []
    for scene_id, run_id, n in scene_run_lengths:
        for _ in range(n):
            rows.append({
                "scene_id": scene_id,
                "run_id": run_id,
                "amplitude": 0.0,
                "present": True,
            })
    df = pd.DataFrame(rows)
    df["amplitude"] = df["amplitude"].astype("float32")
    df["present"] = df["present"].astype(bool)
    df["scene_id"] = df["scene_id"].astype("int64")
    df["run_id"] = df["run_id"].astype("int64")
    df.to_parquet(path, index=False)


class TestExtractRuns:
    def test_single_run_one_window(self, tmp_path):
        # 8 samples, window_size=4 → 2 windows
        p = tmp_path / "single.parquet"
        _write_split_runs_parquet(p, [(1, 6, 8)])
        runs = extract_runs(p, window_size=4)
        assert runs == {(1, 6): (0, 2)}

    def test_multiple_runs_window_aligned(self, tmp_path):
        # window_size=4. Runs of 8, 8, 8 samples → windows (0,2), (2,4), (4,6)
        p = tmp_path / "multi.parquet"
        _write_split_runs_parquet(p, [(1, 6, 8), (1, 7, 8), (2, 6, 8)])
        runs = extract_runs(p, window_size=4)
        assert runs == {
            (1, 6): (0, 2),
            (1, 7): (2, 4),
            (2, 6): (4, 6),
        }

    def test_run_boundary_straddling_window_dropped(self, tmp_path):
        # window_size=4. Run lengths 6, 6 → samples [0,6), [6,12).
        # Run 1 occupies samples 0..5, window 0 = [0,4) is wholly inside,
        #   window 1 = [4,8) straddles. start=ceil(0/4)=0, end=floor(6/4)=1.
        # Run 2 occupies samples 6..11. start=ceil(6/4)=2, end=floor(12/4)=3.
        p = tmp_path / "straddle.parquet"
        _write_split_runs_parquet(p, [(1, 6, 6), (1, 7, 6)])
        runs = extract_runs(p, window_size=4)
        assert runs == {
            (1, 6): (0, 1),
            (1, 7): (2, 3),
        }

    def test_run_too_short_drops_to_empty(self, tmp_path):
        # window_size=4.
        # Run (1,6): 3 samples [0,3) → start=ceil(0/4)=0, end=floor(3/4)=0 → (0,0) empty
        # Run (1,7): 8 samples starting at sample 3, range [3,11)
        #   → start=ceil(3/4)=1, end=floor(11/4)=2 → (1,2)
        p = tmp_path / "tiny.parquet"
        _write_split_runs_parquet(p, [(1, 6, 3), (1, 7, 8)])
        runs = extract_runs(p, window_size=4)
        assert (1, 6) in runs
        assert runs[(1, 6)] == (0, 0)   # empty range, but the key is still returned
        assert runs[(1, 7)] == (1, 2)

    def test_missing_columns_raises(self, tmp_path):
        p = tmp_path / "noruns.parquet"
        df = pd.DataFrame({"amplitude": [0.0] * 8, "present": [True] * 8})
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required column"):
            extract_runs(p, window_size=4)

    def test_non_contiguous_raises(self, tmp_path):
        # Same (scene, run) appears twice non-adjacently
        p = tmp_path / "non_contig.parquet"
        _write_split_runs_parquet(p, [(1, 6, 4), (1, 7, 4), (1, 6, 4)])
        with pytest.raises(ValueError, match="non-contiguous"):
            extract_runs(p, window_size=4)


from crl_vehicle.data.id_split import pair_runs


class TestPairRuns:
    def test_simple_intersection(self):
        audio   = {(1, 6): (0, 100), (1, 7): (100, 150)}
        seismic = {(1, 6): (0, 100), (1, 7): (100, 150)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(1, 6): (0, 100), (1, 7): (100, 150)}
        assert dropped == []

    def test_intersection_smaller_than_either(self):
        audio   = {(1, 6): (0, 100)}
        seismic = {(1, 6): (10, 90)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(1, 6): (10, 90)}
        assert dropped == []

    def test_run_only_in_audio_dropped_with_reason(self):
        audio   = {(1, 6): (0, 100), (1, 7): (100, 150)}
        seismic = {(1, 6): (0, 100)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(1, 6): (0, 100)}
        assert dropped == [{"run_key": (1, 7), "reason": "single_sensor"}]

    def test_run_only_in_seismic_dropped(self):
        audio   = {(1, 6): (0, 100)}
        seismic = {(1, 6): (0, 100), (2, 7): (100, 200)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(1, 6): (0, 100)}
        assert dropped == [{"run_key": (2, 7), "reason": "single_sensor"}]

    def test_empty_intersection_dropped(self):
        # Audio range [0, 50), seismic range [50, 100) → intersection empty
        audio   = {(1, 6): (0, 50)}
        seismic = {(1, 6): (50, 100)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {}
        assert dropped == [{"run_key": (1, 6), "reason": "empty_intersection"}]

    def test_drop_does_not_cascade_across_scenes(self):
        # run_id=6 has empty intersection in scene 1 but valid in scene 2
        audio   = {(1, 6): (0, 50), (2, 6): (100, 200)}
        seismic = {(1, 6): (50, 100), (2, 6): (100, 200)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(2, 6): (100, 200)}
        assert dropped == [{"run_key": (1, 6), "reason": "empty_intersection"}]

    def test_drop_does_not_cascade_across_run_ids(self):
        # scene_id=1 has run_id=6 empty, but run_id=7 is valid
        audio   = {(1, 6): (0, 50), (1, 7): (100, 200)}
        seismic = {(1, 6): (50, 100), (1, 7): (100, 200)}
        paired, dropped = pair_runs(audio, seismic)
        assert paired == {(1, 7): (100, 200)}
        assert dropped == [{"run_key": (1, 6), "reason": "empty_intersection"}]
