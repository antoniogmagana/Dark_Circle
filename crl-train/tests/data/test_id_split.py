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


from crl_vehicle.data.id_split import partition_runs_50_25_25


class TestPartitionRuns502525:
    def test_no_runs_returns_empty(self):
        assert partition_runs_50_25_25({}) == {}

    def test_one_run_goes_to_train(self):
        result = partition_runs_50_25_25({(1, 6): (0, 100)})
        assert result == {(1, 6): "train"}

    def test_two_runs_train_val(self):
        # Largest first → train; next → val
        result = partition_runs_50_25_25({(1, 6): (0, 100), (1, 7): (100, 160)})
        assert result == {(1, 6): "train", (1, 7): "val"}

    def test_three_equal_runs_floor_enforced(self):
        # 3 runs of 100 each. Greedy: train, val, test (floor satisfied).
        runs = {(1, 6): (0, 100), (1, 7): (100, 200), (2, 6): (200, 300)}
        result = partition_runs_50_25_25(runs)
        assert set(result.values()) == {"train", "val", "test"}

    def test_eight_uneven_runs_close_to_50_25_25(self):
        # Emulates the inspected cx30_rs1: 8 runs of varied size.
        runs = {
            (1, 6): (0, 181),
            (1, 7): (181, 241),
            (2, 2): (241, 422),
            (2, 3): (422, 485),
            (4, 6): (485, 575),
            (4, 7): (575, 773),
            (5, 6): (773, 957),
            (5, 7): (957, 1019),
        }
        result = partition_runs_50_25_25(runs)
        # Total paired windows: 1019. Targets: train=509, val=255, test=255.
        totals = {"train": 0, "val": 0, "test": 0}
        for key, split in result.items():
            s, e = runs[key]
            totals[split] += e - s
        # Within ±15% of target ratios is fine.
        assert 0.40 <= totals["train"] / 1019 <= 0.60
        assert 0.15 <= totals["val"]   / 1019 <= 0.35
        assert 0.15 <= totals["test"]  / 1019 <= 0.35
        # All three buckets must be non-empty (floor)
        assert totals["val"]  > 0
        assert totals["test"] > 0

    def test_deterministic_across_calls(self):
        runs = {(1, 6): (0, 100), (1, 7): (100, 200), (2, 6): (200, 300)}
        a = partition_runs_50_25_25(runs)
        b = partition_runs_50_25_25(runs)
        assert a == b

    def test_floor_when_greedy_would_starve_val(self):
        # 3 runs but one is enormous. Without floor, greedy might give
        # train both small ones. With floor, val and test each get ≥1.
        runs = {(1, 6): (0, 1000), (1, 7): (1000, 1010), (2, 6): (1010, 1020)}
        result = partition_runs_50_25_25(runs)
        splits = set(result.values())
        assert "val"  in splits
        assert "test" in splits


from crl_vehicle.data.id_split import compute_manifest_hash, build_manifest


class TestComputeManifestHash:
    def test_returns_64char_hex(self, tmp_path):
        p = tmp_path / "f.parquet"
        p.write_text("dummy")
        h = compute_manifest_hash(
            mapping={"iobt": {"x": ["light", "y", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            source_files=[("iobt_audio_x_rs1", p)],
        )
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_inputs_same_hash(self, tmp_path):
        p = tmp_path / "f.parquet"
        p.write_text("dummy")
        kwargs = dict(
            mapping={"iobt": {"x": ["light", "y", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            source_files=[("iobt_audio_x_rs1", p)],
        )
        assert compute_manifest_hash(**kwargs) == compute_manifest_hash(**kwargs)

    def test_changed_mapping_changes_hash(self, tmp_path):
        p = tmp_path / "f.parquet"
        p.write_text("dummy")
        common = dict(
            window_sizes={"audio": 16000, "seismic": 200},
            source_files=[("iobt_audio_x_rs1", p)],
        )
        h1 = compute_manifest_hash(
            mapping={"iobt": {"x": ["light", "y", "split"]}}, **common,
        )
        h2 = compute_manifest_hash(
            mapping={"iobt": {"x": ["light", "y", "split_runs"]}}, **common,
        )
        assert h1 != h2

    def test_changed_window_size_changes_hash(self, tmp_path):
        p = tmp_path / "f.parquet"
        p.write_text("dummy")
        common = dict(
            mapping={"iobt": {"x": ["light", "y", "split"]}},
            source_files=[("iobt_audio_x_rs1", p)],
        )
        h1 = compute_manifest_hash(
            window_sizes={"audio": 16000, "seismic": 200}, **common,
        )
        h2 = compute_manifest_hash(
            window_sizes={"audio": 16000, "seismic": 400}, **common,
        )
        assert h1 != h2

    def test_changed_mtime_changes_hash(self, tmp_path):
        import os, time
        p = tmp_path / "f.parquet"
        p.write_text("v1")
        common = dict(
            mapping={"iobt": {"x": ["light", "y", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
        )
        h1 = compute_manifest_hash(source_files=[("iobt_audio_x_rs1", p)], **common)
        # Touch with a future mtime
        os.utime(p, (time.time() + 100, time.time() + 100))
        h2 = compute_manifest_hash(source_files=[("iobt_audio_x_rs1", p)], **common)
        assert h1 != h2

    def test_source_file_order_invariant(self, tmp_path):
        a = tmp_path / "a.parquet"; a.write_text("a")
        b = tmp_path / "b.parquet"; b.write_text("b")
        common = dict(
            mapping={"iobt": {"x": ["light", "y", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
        )
        h1 = compute_manifest_hash(source_files=[("a", a), ("b", b)], **common)
        h2 = compute_manifest_hash(source_files=[("b", b), ("a", a)], **common)
        assert h1 == h2


import logging


def _write_simple_parquet(path: Path, n_samples: int) -> None:
    """Write a parquet with amplitude/present (no scene_id/run_id)."""
    df = pd.DataFrame({
        "amplitude": np.zeros(n_samples, dtype="float32"),
        "present":   np.ones(n_samples, dtype=bool),
    })
    df.to_parquet(path, index=False)


class TestBuildManifest:
    def test_split_marker_produces_val_test_assignments(self, tmp_path):
        # Set up: one "split" file in iobt
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        # 10 windows of audio (window_size=16000) and 10 of seismic (200)
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)

        mapping = {"iobt": {"silverado": ["heavy", "pickup", "split"]}}
        manifest = build_manifest(
            id_root=tmp_path,
            mapping=mapping,
            window_sizes={"audio": 16000, "seismic": 200},
        )
        assert "groups" in manifest
        gkey = "iobt__silverado__rs1"
        assert gkey in manifest["groups"]
        g = manifest["groups"][gkey]
        assert g["marker"] == "split"
        assert g["split_assignments"] == {
            "val":  [[0, 5]],
            "test": [[5, 10]],
        }

    def test_split_marker_only_one_sensor_uses_that_count(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        # Only audio, no seismic → N = audio_n_windows = 8
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=128_000)
        mapping = {"iobt": {"silverado": ["heavy", "pickup", "split"]}}
        manifest = build_manifest(
            id_root=tmp_path,
            mapping=mapping,
            window_sizes={"audio": 16000, "seismic": 200},
        )
        g = manifest["groups"]["iobt__silverado__rs1"]
        assert g["split_assignments"] == {
            "val":  [[0, 4]],
            "test": [[4, 8]],
        }

    def test_split_marker_too_few_windows_skipped(self, tmp_path, caplog):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        # 1 window only (16000 samples) → N_pair = 1, can't split
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=16_000)
        mapping = {"iobt": {"silverado": ["heavy", "pickup", "split"]}}
        with caplog.at_level(logging.INFO):
            manifest = build_manifest(
                id_root=tmp_path,
                mapping=mapping,
                window_sizes={"audio": 16000, "seismic": 200},
            )
        assert "iobt__silverado__rs1" not in manifest["groups"]
        assert any("too few windows" in r.message.lower() or
                   "skipping" in r.message.lower() for r in caplog.records)

    def test_train_val_test_markers_no_routing_computation(self, tmp_path):
        # "train" / "val" / "test" markers don't produce manifest groups —
        # they're handled at index-build time, not at manifest time.
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _write_simple_parquet(train_dir / "iobt_audio_polaris_rs1.parquet",
                              n_samples=160_000)
        mapping = {"iobt": {"polaris": ["light", "polaris", "train"]}}
        manifest = build_manifest(
            id_root=tmp_path, mapping=mapping,
            window_sizes={"audio": 16000, "seismic": 200},
        )
        # No group entry needed for plain train/val/test markers
        assert manifest["groups"] == {}

    def test_dedupes_files_across_subdirs(self, tmp_path):
        # Same file present in train/ and val/ → only one group entry
        for sub in ("train", "val"):
            d = tmp_path / sub
            d.mkdir()
            _write_simple_parquet(d / "iobt_audio_silverado_rs1.parquet",
                                  n_samples=160_000)
            _write_simple_parquet(d / "iobt_seismic_silverado_rs1.parquet",
                                  n_samples=2_000)
        mapping = {"iobt": {"silverado": ["heavy", "pickup", "split"]}}
        manifest = build_manifest(
            id_root=tmp_path, mapping=mapping,
            window_sizes={"audio": 16000, "seismic": 200},
        )
        # Exactly one group, regardless of which subdir we picked from
        assert len(manifest["groups"]) == 1

    def test_manifest_includes_metadata(self, tmp_path):
        train_dir = tmp_path / "train"; train_dir.mkdir()
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)
        mapping = {"iobt": {"silverado": ["heavy", "pickup", "split"]}}
        manifest = build_manifest(
            id_root=tmp_path, mapping=mapping,
            window_sizes={"audio": 16000, "seismic": 200},
        )
        assert manifest["schema_version"] == 1
        assert manifest["config_window_sizes"] == {"audio": 16000, "seismic": 200}
        assert "created_unix" in manifest
