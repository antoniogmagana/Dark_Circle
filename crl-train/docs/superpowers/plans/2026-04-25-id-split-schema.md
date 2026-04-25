# In-Distribution Split Schema (`--use-id-split`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--use-id-split` schema that derives train/val/test from `DATASET_VEHICLE_MAP` rather than directory layout, supporting `"split"` (whole-window half/half) and `"split_runs"` (paired-run 50/25/25) markers, with a manifest cache under `saved_crl/id_cache/`.

**Architecture:** Source-of-truth flips per flag — when off, dir layout drives splits (today's behavior, untouched); when on, the marker triple in `DATASET_VEHICLE_MAP` drives splits and the loader scans all three of `parsed/{train,val,test}/` together. New `id_split` module computes per-group window-range manifests, keyed by `(dataset, vehicle, rs_node)` so audio/seismic share paired-window coordinates and cannot drift. `SensorDataset._build_from_parquet` gets one routing branch; everything downstream (`.pt` cache, `_get_window`, `StratifiedPairDataset`) is unchanged.

**Tech Stack:** Python 3, PyTorch, PyArrow, pandas (test fixtures), pytest. Existing project conventions: dataclass config, `tmp_path` pytest fixtures, no mocking of disk for IO tests.

**Spec:** [`docs/superpowers/specs/2026-04-25-id-split-schema-design.md`](../specs/2026-04-25-id-split-schema-design.md)

---

## File Structure

**New files:**
- `crl-train/crl_vehicle/data/id_split.py` — manifest builder, hash logic, `"split"` and `"split_runs"` algorithms. One responsibility: turn parquet sources + mapping into a per-group window-range manifest. Pure functions where possible; no torch/CUDA dependencies.
- `crl-train/tests/data/test_id_split.py` — unit tests for the manifest builder.
- `crl-train/tests/data/test_id_split_dataset.py` — integration tests: `SensorDataset` constructed with `use_id_split=True` against synthetic parquet fixtures.

**Modified files:**
- `crl-train/crl_vehicle/config.py` — add `use_id_split: bool = False` to `CRLConfig` (cosmetic; CLI flag is the real switch).
- `crl-train/crl_vehicle/data/dataset.py` — add `use_id_split` / `role` / `id_root` / `id_cache_dir` kwargs to `SensorDataset.__init__`, branch in `_load_data` / `_build_from_parquet`. No changes to `StratifiedPairDataset`, `_preload_shared`, `_get_window`, presence logic, or collate functions.
- `crl-train/train.py` — add `--use-id-split` and `--id-root` CLI flags + dataset wiring.

**Created at runtime:**
- `crl-train/saved_crl/id_cache/` — manifest cache directory (created on first use).

---

## Task 1: Add `use_id_split` config field

**Files:**
- Modify: `crl-train/crl_vehicle/config.py`
- Test: `crl-train/tests/data/test_id_split.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `crl-train/tests/data/test_id_split.py`:

```python
"""Unit tests for ID split manifest builder."""
from crl_vehicle.config import CRLConfig


def test_use_id_split_field_defaults_false():
    cfg = CRLConfig()
    assert cfg.use_id_split is False


def test_use_id_split_field_settable():
    cfg = CRLConfig(use_id_split=True)
    assert cfg.use_id_split is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'use_id_split'` (and the first test fails on `AttributeError`).

- [ ] **Step 3: Add the field**

In `crl-train/crl_vehicle/config.py`, locate the `# Data` section in `CRLConfig` (around line 149–155) and add:

```python
    # ID split schema (opt-in). When True, train.py reads
    # train/val/test assignments from DATASET_VEHICLE_MAP rather than
    # from on-disk directory layout. See
    # docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
    use_id_split: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v`
Expected: PASS, both tests green.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/config.py crl-train/tests/data/test_id_split.py
git commit -m "feat(config): add use_id_split flag to CRLConfig"
```

---

## Task 2: Pure helper — `compute_split_intervals` for `"split"` marker

**Files:**
- Create: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

This task implements the simplest case: half-and-half on a paired window count, val gets first half, test gets second half. No file IO yet.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
import pytest
from crl_vehicle.data.id_split import compute_split_intervals


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crl_vehicle.data.id_split'`.

- [ ] **Step 3: Create the module with the helper**

Create `crl-train/crl_vehicle/data/id_split.py`:

```python
"""Manifest builder for the ID split schema (--use-id-split).

Computes per-group window-range manifests from DATASET_VEHICLE_MAP
markers ("split", "split_runs"). Designed to be pure (no torch, no
CUDA) and cache-friendly.

See docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
"""
from __future__ import annotations


def compute_split_intervals(n_paired: int) -> dict[str, list[tuple[int, int]]] | None:
    """Half/half split on paired window count for "split" marker.

    Returns:
        {"val": [(0, n_paired // 2)], "test": [(n_paired // 2, n_paired)]}
        or None if n_paired < 2 (cannot split).

    Intervals are half-open [start, end) in paired-window coordinates.
    """
    if n_paired < 2:
        return None
    half = n_paired // 2
    return {
        "val":  [(0, half)],
        "test": [(half, n_paired)],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add compute_split_intervals for split marker"
```

---

## Task 3: Pure helper — `extract_runs` from `(scene_id, run_id)` columns

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

Detect `(scene_id, run_id)` blocks in a single sensor's parquet, return per-run window ranges. Verifies contiguity and aborts on out-of-order data.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
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
        # 3 samples for window_size=4 → ceil(0/4)=0, floor(3/4)=0 → empty range
        p = tmp_path / "tiny.parquet"
        _write_split_runs_parquet(p, [(1, 6, 3), (1, 7, 8)])
        runs = extract_runs(p, window_size=4)
        # Empty-range runs ARE returned (range start==end). Caller decides
        # what to do — they show up as candidates for the pairing rule.
        assert (1, 6) in runs
        assert runs[(1, 6)] == (0, 0)
        assert runs[(1, 7)] == (0, 2)  # starts at sample 3 → ceil(3/4)=1; ends at 11 → floor(11/4)=2 → (1,2)? recompute

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
```

Note on `test_run_too_short_drops_to_empty`: the (1,7) start computation is delicate — recompute it: samples 3..10 (8 samples starting at sample 3), window_size=4 → start_window=ceil(3/4)=1, end_window=floor(11/4)=2 → range (1, 2). Update the assertion to:

```python
        assert runs[(1, 7)] == (1, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k extract_runs`
Expected: FAIL with `ImportError` for `extract_runs`.

- [ ] **Step 3: Implement `extract_runs`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def extract_runs(
    parquet_path: Path,
    window_size: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Extract per-run window ranges from a parquet with scene_id/run_id.

    Args:
        parquet_path: parquet with columns 'scene_id' (int64), 'run_id' (int64).
        window_size: samples per window (e.g. 16000 for audio, 200 for seismic).

    Returns:
        {(scene_id, run_id): (start_window, end_window)} where ranges are
        half-open [start, end) and use ceil(start_sample / W),
        floor(end_sample / W) to drop windows that straddle a run boundary.

    Raises:
        ValueError: if scene_id or run_id columns are missing, or if any
            (scene, run) key is non-contiguous in the file.
    """
    parquet_path = Path(parquet_path)
    table = pq.read_table(parquet_path, columns=["scene_id", "run_id"])
    cols = set(table.column_names)
    for col in ("scene_id", "run_id"):
        if col not in cols:
            raise ValueError(
                f"{parquet_path.name}: missing required column {col!r} "
                f"for split_runs marker"
            )

    scene = table.column("scene_id").to_numpy()
    run   = table.column("run_id").to_numpy()
    n     = len(scene)
    if n == 0:
        return {}

    # Find block boundaries: positions where (scene, run) changes.
    key_changed = (scene[1:] != scene[:-1]) | (run[1:] != run[:-1])
    boundaries  = np.flatnonzero(key_changed) + 1  # start indices of new blocks
    block_starts = np.concatenate(([0], boundaries))
    block_ends   = np.concatenate((boundaries, [n]))

    # Verify contiguity: each (scene, run) appears in exactly one block
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for s, e in zip(block_starts, block_ends):
        key = (int(scene[s]), int(run[s]))
        if key in seen:
            raise ValueError(
                f"{parquet_path.name}: (scene_id={key[0]}, run_id={key[1]}) "
                f"appears in non-contiguous blocks (sample ranges "
                f"[{seen[key][0]},{seen[key][1]}) and [{s},{e}))"
            )
        seen[key] = (int(s), int(e))

    # Convert sample ranges to window ranges with ceil/floor
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (s, e) in seen.items():
        w_start = math.ceil(s / window_size)
        w_end   = e // window_size  # floor of e
        out[key] = (w_start, w_end)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k extract_runs`
Expected: PASS, all `extract_runs` tests green.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add extract_runs for split_runs marker"
```

---

## Task 4: Pure helper — `pair_runs` (intersect audio + seismic per `(scene, run)`)

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

For each `(scene, run)` key in either sensor: if it's in both AND the intersection is non-empty, keep the paired range; else drop and record the reason. **Drops are scoped to one `(scene, run)` only.**

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k pair_runs`
Expected: FAIL with `ImportError` for `pair_runs`.

- [ ] **Step 3: Implement `pair_runs`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
def pair_runs(
    audio_runs:   dict[tuple[int, int], tuple[int, int]],
    seismic_runs: dict[tuple[int, int], tuple[int, int]],
) -> tuple[
    dict[tuple[int, int], tuple[int, int]],
    list[dict],
]:
    """Intersect per-sensor run ranges per (scene, run) key.

    Returns:
        (paired, dropped):
          paired:  {(scene, run): (start_w, end_w)} for keys present in both
                   sensors with non-empty intersection.
          dropped: [{"run_key": (scene, run), "reason": str}, ...]

    Drops are per-(scene, run) only — never cascade across scenes or run_ids.
    Dropped list is sorted deterministically by run_key for reproducibility.
    """
    paired: dict[tuple[int, int], tuple[int, int]] = {}
    dropped: list[dict] = []

    all_keys = set(audio_runs) | set(seismic_runs)
    for key in sorted(all_keys):
        a = audio_runs.get(key)
        s = seismic_runs.get(key)
        if a is None or s is None:
            dropped.append({"run_key": key, "reason": "single_sensor"})
            continue
        start = max(a[0], s[0])
        end   = min(a[1], s[1])
        if start >= end:
            dropped.append({"run_key": key, "reason": "empty_intersection"})
            continue
        paired[key] = (start, end)
    return paired, dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k pair_runs`
Expected: PASS, all `pair_runs` tests green.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add pair_runs with per-(scene,run) drop scoping"
```

---

## Task 5: Pure helper — `partition_runs_50_25_25` greedy assignment

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

Process surviving paired runs in descending paired-window-count order; assign each to whichever bucket is currently furthest below its target ratio. Tie-break: train > val > test. Floor: if ≥3 keys, val and test each get ≥1.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k partition`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `partition_runs_50_25_25`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
_TARGET_RATIOS = {"train": 0.50, "val": 0.25, "test": 0.25}
_TIE_ORDER = ("train", "val", "test")


def partition_runs_50_25_25(
    paired_runs: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Greedy 50/25/25 partition of paired runs over window counts.

    Args:
        paired_runs: {(scene, run): (start_w, end_w)} from pair_runs().

    Returns:
        {(scene, run): "train" | "val" | "test"}.

    Algorithm:
      - Sort runs by descending paired-window-count, ties by (scene, run).
      - For each run, assign to the bucket whose deficit (target - current)
        is largest. Ties resolved by _TIE_ORDER (train > val > test).
      - Floor: if ≥3 surviving runs, val and test each receive ≥1 (swap
        the smallest train run into the empty bucket if needed).
    """
    if not paired_runs:
        return {}

    # Descending by length, then ascending by key for determinism
    items = sorted(
        paired_runs.items(),
        key=lambda kv: (-(kv[1][1] - kv[1][0]), kv[0]),
    )
    total = sum(e - s for _, (s, e) in items)

    assignment: dict[tuple[int, int], str] = {}
    bucket_totals = {"train": 0, "val": 0, "test": 0}

    for key, (s, e) in items:
        n = e - s
        # Pick bucket with largest deficit; break ties by _TIE_ORDER
        deficits = {
            b: _TARGET_RATIOS[b] * total - bucket_totals[b]
            for b in _TIE_ORDER
        }
        max_deficit = max(deficits.values())
        choice = next(b for b in _TIE_ORDER if deficits[b] == max_deficit)
        assignment[key] = choice
        bucket_totals[choice] += n

    # Floor: with ≥3 runs, val and test must each have ≥1
    if len(items) >= 3:
        for needy in ("val", "test"):
            if not any(v == needy for v in assignment.values()):
                # Swap the smallest train run (by paired-window count) into needy
                train_keys = [k for k, v in assignment.items() if v == "train"]
                if train_keys:
                    smallest_train = min(
                        train_keys,
                        key=lambda k: paired_runs[k][1] - paired_runs[k][0],
                    )
                    assignment[smallest_train] = needy
    return assignment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k partition`
Expected: PASS, all partition tests green.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add partition_runs_50_25_25 greedy with floor"
```

---

## Task 6: Manifest hash — `compute_manifest_hash`

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

The hash makes the cache self-invalidating: any change to mapping, window sizes, or source mtimes produces a new hash.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
from crl_vehicle.data.id_split import compute_manifest_hash


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k manifest_hash`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `compute_manifest_hash`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
import hashlib
import json
from pathlib import Path


def compute_manifest_hash(
    mapping: dict,
    window_sizes: dict[str, int],
    source_files: list[tuple[str, Path]],
) -> str:
    """SHA-256 over (mapping, window sizes, sorted source-file mtimes_ns).

    Args:
        mapping: DATASET_VEHICLE_MAP (or a subset).
        window_sizes: {"audio": int, "seismic": int}.
        source_files: list of (stem, parquet_path) for files whose
            routing depends on per-file computation (split / split_runs).

    Returns:
        Hex SHA-256 digest (64 chars).
    """
    # Sort source files by stem for order invariance
    sources_sorted = sorted(source_files, key=lambda kv: kv[0])
    mtimes = [
        (stem, Path(p).stat().st_mtime_ns) for stem, p in sources_sorted
    ]

    payload = {
        "mapping":      mapping,
        "window_sizes": window_sizes,
        "sources":      mtimes,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k manifest_hash`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add compute_manifest_hash for cache invalidation"
```

---

## Task 7: Manifest builder — `build_manifest`

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

Top-level entry point: walk a parquet root, group files by `(dataset, vehicle, rs)`, look up the mapping marker, dispatch to the appropriate algorithm, return the full manifest dict.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
import logging
from crl_vehicle.data.id_split import build_manifest


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
            "val":  [(0, 5)],
            "test": [(5, 10)],
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
            "val":  [(0, 4)],
            "test": [(4, 8)],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k build_manifest`
Expected: FAIL with `ImportError` for `build_manifest`.

- [ ] **Step 3: Implement `build_manifest`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
import logging
import time

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_KNOWN_DATASETS = {"iobt", "focal", "m3nvc"}


def _parse_stem(stem: str, sensor: str) -> tuple[str, str, str] | None:
    """Parse '{dataset}_{sensor}_{vehicle}_{rs}' → (dataset, vehicle, rs)."""
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] not in _KNOWN_DATASETS or parts[1] != sensor:
        return None
    rs = parts[-1]
    if not rs.startswith("rs"):
        return None
    return parts[0], "_".join(parts[2:-1]), rs


def _group_key(dataset: str, vehicle: str, rs: str) -> str:
    return f"{dataset}__{vehicle}__{rs}"


def _file_n_windows(path: Path, window_size: int) -> int:
    return pq.read_metadata(path).num_rows // window_size


def build_manifest(
    id_root: Path,
    mapping: dict,
    window_sizes: dict[str, int],
) -> dict:
    """Walk id_root, build per-group ID-split manifest.

    Scans id_root/*/*.parquet, dedupes by stem, and for every group
    whose marker is "split" or "split_runs", computes split intervals
    in paired-window coordinates.

    "train"/"val"/"test" markers produce no manifest entry — those
    files are routed at index-build time, not via the manifest.

    Args:
        id_root: parent containing train/, val/, test/ subdirs.
        mapping: DATASET_VEHICLE_MAP-shaped dict.
        window_sizes: {"audio": int, "seismic": int}.

    Returns:
        Manifest dict matching the schema in the design spec.
    """
    id_root = Path(id_root)
    # Scan all subdirs, dedupe by stem
    all_files: dict[str, Path] = {}
    for parquet in id_root.glob("*/*.parquet"):
        all_files.setdefault(parquet.stem, parquet)

    # Group by (dataset, vehicle, rs)
    audio_files:   dict[tuple[str, str, str], Path] = {}
    seismic_files: dict[tuple[str, str, str], Path] = {}
    for stem, path in all_files.items():
        for sensor, dest in (("audio", audio_files), ("seismic", seismic_files)):
            parsed = _parse_stem(stem, sensor)
            if parsed is not None:
                dest[parsed] = path
                break

    groups: dict[str, dict] = {}
    all_keys = set(audio_files) | set(seismic_files)

    for ds, vehicle, rs in sorted(all_keys):
        ds_map = mapping.get(ds, {})
        entry  = ds_map.get(vehicle)
        if entry is None or len(entry) < 3:
            # Background or unknown — no manifest entry needed
            continue
        marker = entry[2]
        if marker not in ("split", "split_runs"):
            continue  # plain train/val/test handled at index time

        gkey = _group_key(ds, vehicle, rs)
        a_path = audio_files.get((ds, vehicle, rs))
        s_path = seismic_files.get((ds, vehicle, rs))

        if marker == "split":
            audio_nw   = _file_n_windows(a_path, window_sizes["audio"])     if a_path else 0
            seismic_nw = _file_n_windows(s_path, window_sizes["seismic"])   if s_path else 0
            if audio_nw and seismic_nw:
                n_paired = min(audio_nw, seismic_nw)
            else:
                n_paired = audio_nw or seismic_nw
            intervals = compute_split_intervals(n_paired)
            if intervals is None:
                logger.info(
                    f"id_split: skipping group {gkey!r} — too few windows "
                    f"for split (n_paired={n_paired})"
                )
                continue
            groups[gkey] = {
                "dataset": ds, "vehicle": vehicle, "rs_node": rs,
                "marker": marker,
                "split_assignments": {
                    k: [list(iv) for iv in v] for k, v in intervals.items()
                },
            }

        elif marker == "split_runs":
            if a_path is None or s_path is None:
                logger.info(
                    f"id_split: skipping group {gkey!r} — split_runs requires "
                    f"both audio and seismic (have audio={a_path is not None}, "
                    f"seismic={s_path is not None})"
                )
                continue
            audio_runs   = extract_runs(a_path, window_sizes["audio"])
            seismic_runs = extract_runs(s_path, window_sizes["seismic"])
            paired, dropped = pair_runs(audio_runs, seismic_runs)
            for d in dropped:
                logger.warning(
                    f"id_split: dropped run {d['run_key']} from group {gkey!r} "
                    f"(reason={d['reason']})"
                )
            if not paired:
                logger.info(
                    f"id_split: skipping group {gkey!r} — no surviving "
                    f"(scene, run) keys after pairing"
                )
                continue
            assignment = partition_runs_50_25_25(paired)
            split_assignments: dict[str, list[list[int]]] = {
                "train": [], "val": [], "test": [],
            }
            run_meta: dict[str, dict] = {}
            for run_key, split in assignment.items():
                start, end = paired[run_key]
                split_assignments[split].append([int(start), int(end)])
                run_meta[f"{run_key[0]}_{run_key[1]}"] = {
                    "split": split,
                    "n_windows_paired": int(end - start),
                }
            # Sort intervals within each split for reproducibility
            for s in split_assignments:
                split_assignments[s].sort()
            groups[gkey] = {
                "dataset": ds, "vehicle": vehicle, "rs_node": rs,
                "marker": marker,
                "split_assignments": split_assignments,
                "run_meta": run_meta,
                "dropped_runs": [
                    {"run_key": f"{d['run_key'][0]}_{d['run_key'][1]}",
                     "reason": d["reason"]}
                    for d in dropped
                ],
            }

    return {
        "schema_version": 1,
        "created_unix": int(time.time()),
        "config_window_sizes": dict(window_sizes),
        "groups": groups,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k build_manifest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add build_manifest dispatching split/split_runs"
```

---

## Task 8: Cache load/save — `load_or_build_manifest`

**Files:**
- Modify: `crl-train/crl_vehicle/data/id_split.py`
- Modify: `crl-train/tests/data/test_id_split.py`

Wraps `build_manifest` with disk-caching keyed by hash. On cache miss (or corrupt cache file), rebuild and atomically write.

- [ ] **Step 1: Write the failing tests**

Append to `crl-train/tests/data/test_id_split.py`:

```python
from crl_vehicle.data.id_split import load_or_build_manifest


class TestLoadOrBuildManifest:
    def test_first_call_writes_manifest(self, tmp_path):
        train_dir = tmp_path / "data" / "train"; train_dir.mkdir(parents=True)
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)

        cache_dir = tmp_path / "id_cache"
        manifest = load_or_build_manifest(
            id_root=tmp_path / "data",
            mapping={"iobt": {"silverado": ["heavy", "pickup", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            cache_dir=cache_dir,
        )
        assert "iobt__silverado__rs1" in manifest["groups"]
        # Cache file written
        cache_files = list(cache_dir.glob("manifest_*.json"))
        assert len(cache_files) == 1

    def test_second_call_hits_cache(self, tmp_path):
        train_dir = tmp_path / "data" / "train"; train_dir.mkdir(parents=True)
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)

        cache_dir = tmp_path / "id_cache"
        kwargs = dict(
            id_root=tmp_path / "data",
            mapping={"iobt": {"silverado": ["heavy", "pickup", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            cache_dir=cache_dir,
        )
        m1 = load_or_build_manifest(**kwargs)
        # Tamper with manifest to detect cache hit
        cache_file = next(cache_dir.glob("manifest_*.json"))
        import json as _json
        data = _json.loads(cache_file.read_text())
        data["sentinel"] = "i was here"
        cache_file.write_text(_json.dumps(data))

        m2 = load_or_build_manifest(**kwargs)
        assert m2.get("sentinel") == "i was here"

    def test_changed_mapping_invalidates_cache(self, tmp_path):
        train_dir = tmp_path / "data" / "train"; train_dir.mkdir(parents=True)
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)
        cache_dir = tmp_path / "id_cache"

        load_or_build_manifest(
            id_root=tmp_path / "data",
            mapping={"iobt": {"silverado": ["heavy", "pickup", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            cache_dir=cache_dir,
        )
        # Different mapping → new hash → new file
        load_or_build_manifest(
            id_root=tmp_path / "data",
            mapping={"iobt": {"silverado": ["heavy", "pickup", "train"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            cache_dir=cache_dir,
        )
        cache_files = list(cache_dir.glob("manifest_*.json"))
        assert len(cache_files) == 2

    def test_corrupt_cache_recomputes(self, tmp_path):
        train_dir = tmp_path / "data" / "train"; train_dir.mkdir(parents=True)
        _write_simple_parquet(train_dir / "iobt_audio_silverado_rs1.parquet",
                              n_samples=160_000)
        _write_simple_parquet(train_dir / "iobt_seismic_silverado_rs1.parquet",
                              n_samples=2_000)
        cache_dir = tmp_path / "id_cache"
        kwargs = dict(
            id_root=tmp_path / "data",
            mapping={"iobt": {"silverado": ["heavy", "pickup", "split"]}},
            window_sizes={"audio": 16000, "seismic": 200},
            cache_dir=cache_dir,
        )
        load_or_build_manifest(**kwargs)
        cache_file = next(cache_dir.glob("manifest_*.json"))
        cache_file.write_text("{not valid json")
        # Should recompute, not crash
        m = load_or_build_manifest(**kwargs)
        assert "iobt__silverado__rs1" in m["groups"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k load_or_build`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `load_or_build_manifest`**

Append to `crl-train/crl_vehicle/data/id_split.py`:

```python
def _collect_split_source_files(id_root: Path, mapping: dict) -> list[tuple[str, Path]]:
    """List (stem, path) for every parquet whose marker is split / split_runs."""
    out: list[tuple[str, Path]] = []
    seen_stems: set[str] = set()
    for parquet in Path(id_root).glob("*/*.parquet"):
        stem = parquet.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        for sensor in ("audio", "seismic"):
            parsed = _parse_stem(stem, sensor)
            if parsed is None:
                continue
            ds, vehicle, _ = parsed
            entry = mapping.get(ds, {}).get(vehicle)
            if entry is not None and len(entry) >= 3 and entry[2] in ("split", "split_runs"):
                out.append((stem, parquet))
            break
    return out


def load_or_build_manifest(
    id_root: Path,
    mapping: dict,
    window_sizes: dict[str, int],
    cache_dir: Path,
) -> dict:
    """Load manifest from cache_dir if hash matches, else build and persist.

    Cache filename is `manifest_<sha256>.json`. On corrupt or missing
    file, recompute and atomically write.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources = _collect_split_source_files(id_root, mapping)
    h = compute_manifest_hash(mapping, window_sizes, sources)
    cache_path = cache_dir / f"manifest_{h}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"id_split: cache file {cache_path.name} corrupt "
                f"({type(e).__name__}); recomputing"
            )

    manifest = build_manifest(id_root, mapping, window_sizes)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(cache_path)  # atomic on POSIX
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split.py -v -k load_or_build`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/id_split.py crl-train/tests/data/test_id_split.py
git commit -m "feat(id_split): add load_or_build_manifest with disk cache"
```

---

## Task 9: `SensorDataset` integration — new constructor kwargs + ID-routing branch

**Files:**
- Modify: `crl-train/crl_vehicle/data/dataset.py`
- Create: `crl-train/tests/data/test_id_split_dataset.py`

Add `use_id_split`, `role`, `id_root`, `id_cache_dir` kwargs and an ID-routing branch in `_load_data` / `_build_from_parquet`. Default-off path is byte-identical to today.

- [ ] **Step 1: Write the failing integration tests**

Create `crl-train/tests/data/test_id_split_dataset.py`:

```python
"""Integration tests: SensorDataset with use_id_split=True."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset


# Smaller test window sizes to keep parquet fixtures tiny
TEST_AUDIO_W   = 16
TEST_SEISMIC_W = 4


@pytest.fixture
def small_cfg():
    """CRLConfig with tiny window sizes so fixtures are fast."""
    cfg = CRLConfig()
    # Override modality_cfg to use small windows
    orig = cfg.modality_cfg
    def patched(sensor):
        from crl_vehicle.config import ModalityConfig
        if sensor == "audio":
            return ModalityConfig(sample_rate=TEST_AUDIO_W, window_size=TEST_AUDIO_W, n_channels=1)
        return ModalityConfig(sample_rate=TEST_SEISMIC_W, window_size=TEST_SEISMIC_W, n_channels=1)
    cfg.modality_cfg = patched
    return cfg


def _write_simple_parquet(path, n_samples):
    df = pd.DataFrame({
        "amplitude": np.zeros(n_samples, dtype="float32"),
        "present":   np.ones(n_samples, dtype=bool),
    })
    df.to_parquet(path, index=False)


def _write_split_runs_parquet(path, scene_run_lengths):
    rows = []
    for s, r, n in scene_run_lengths:
        for _ in range(n):
            rows.append({"scene_id": s, "run_id": r,
                         "amplitude": 0.0, "present": True})
    df = pd.DataFrame(rows)
    df["amplitude"] = df["amplitude"].astype("float32")
    df["present"]   = df["present"].astype(bool)
    df["scene_id"]  = df["scene_id"].astype("int64")
    df["run_id"]    = df["run_id"].astype("int64")
    df.to_parquet(path, index=False)


@pytest.fixture
def id_root_split(tmp_path, monkeypatch):
    """Layout: data/{train,val,test}/iobt_{audio,seismic}_silverado_rs1.parquet
    with marker = "split". 10 windows in each sensor."""
    from crl_vehicle import config as cfg_mod
    monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "iobt",
        {"silverado": ["heavy", "pickup", "split"]})

    data_root = tmp_path / "data"
    for sub in ("train", "val", "test"):
        d = data_root / sub
        d.mkdir(parents=True)
    # Put the actual file in train/ (any subdir works under ID schema)
    _write_simple_parquet(data_root / "train" / "iobt_audio_silverado_rs1.parquet",
                          n_samples=TEST_AUDIO_W * 10)
    _write_simple_parquet(data_root / "train" / "iobt_seismic_silverado_rs1.parquet",
                          n_samples=TEST_SEISMIC_W * 10)
    return data_root


class TestIdSplitDatasetRoles:
    def test_train_role_is_empty_for_split_marker(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "train",   # ignored under id split
            config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="train",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        # "split" files contribute zero windows to train
        assert len(ds) == 0

    def test_val_role_gets_first_half(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "val",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        # 10 windows total, half = 5 → val gets indices 0..4
        assert len(ds) == 5
        # Verify w indices are 0..4
        windows = sorted({entry[1] for entry in ds._index})
        assert windows == [0, 1, 2, 3, 4]

    def test_test_role_gets_second_half(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "test",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="test",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        assert len(ds) == 5
        windows = sorted({entry[1] for entry in ds._index})
        assert windows == [5, 6, 7, 8, 9]

    def test_val_and_test_indices_disjoint(self, id_root_split, tmp_path, small_cfg):
        val_ds = SensorDataset(
            parquet_dir=id_root_split / "val",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        test_ds = SensorDataset(
            parquet_dir=id_root_split / "test",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="test",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        v = {entry[1] for entry in val_ds._index}
        t = {entry[1] for entry in test_ds._index}
        assert v.isdisjoint(t)


class TestIdSplitDefaultOffUnchanged:
    def test_use_id_split_false_does_not_require_new_kwargs(
        self, id_root_split, tmp_path, small_cfg
    ):
        # With the flag off, behavior should match today exactly —
        # _index is built solely from parquet_dir, ignoring DATASET_VEHICLE_MAP marker.
        ds = SensorDataset(
            parquet_dir=id_root_split / "train",
            config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
        )
        # The "split" marker has no effect when use_id_split=False
        assert len(ds) == 10  # all 10 windows from the file


class TestIdSplitTrainMarker:
    def test_train_marker_routes_all_to_train(self, tmp_path, small_cfg, monkeypatch):
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "iobt",
            {"polaris": ["light", "polaris", "train"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        _write_simple_parquet(data_root / "train" / "iobt_audio_polaris_rs1.parquet",
                              n_samples=TEST_AUDIO_W * 6)
        _write_simple_parquet(data_root / "train" / "iobt_seismic_polaris_rs1.parquet",
                              n_samples=TEST_SEISMIC_W * 6)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        train_ds = SensorDataset(
            parquet_dir=data_root / "train", config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="train",
            id_root=data_root, id_cache_dir=tmp_path / "id_cache",
        )
        assert len(train_ds) == 6

        val_ds = SensorDataset(
            parquet_dir=data_root / "val", config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=data_root, id_cache_dir=tmp_path / "id_cache",
        )
        assert len(val_ds) == 0


class TestIdSplitRunsMarker:
    def test_split_runs_partitions_across_three_roles(self, tmp_path, small_cfg, monkeypatch):
        """End-to-end split_runs: 3 paired runs → train/val/test all non-empty."""
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "m3nvc",
            {"cx30": ["medium", "cx30", "split_runs"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        # Three runs, each large enough to yield a few windows after ceil/floor.
        # Audio window=16, seismic window=4. Each run = 64 audio samples + 16 seismic = 4 windows.
        audio_runs   = [(1, 6, TEST_AUDIO_W * 4), (1, 7, TEST_AUDIO_W * 4),
                        (2, 6, TEST_AUDIO_W * 4)]
        seismic_runs = [(1, 6, TEST_SEISMIC_W * 4), (1, 7, TEST_SEISMIC_W * 4),
                        (2, 6, TEST_SEISMIC_W * 4)]
        _write_split_runs_parquet(data_root / "train" / "m3nvc_audio_cx30_rs1.parquet",
                                  audio_runs)
        _write_split_runs_parquet(data_root / "train" / "m3nvc_seismic_cx30_rs1.parquet",
                                  seismic_runs)

        lengths = {}
        for role in ("train", "val", "test"):
            ds = SensorDataset(
                parquet_dir=data_root / role, config=small_cfg, is_train=(role == "train"),
                cache_dir=tmp_path / f"raw_cache_{role}",
                use_id_split=True, role=role,
                id_root=data_root, id_cache_dir=tmp_path / "id_cache",
            )
            lengths[role] = len(ds)
        # All three roles must have ≥1 window (floor)
        assert lengths["train"] >= 1
        assert lengths["val"]   >= 1
        assert lengths["test"]  >= 1
        # Total windows = 12 (3 runs × 4 windows each, with whole-run boundaries)
        # But ceil/floor may shave a few edges — accept 9..12.
        assert 9 <= sum(lengths.values()) <= 12

    def test_split_runs_dropped_run_warning(self, tmp_path, small_cfg, monkeypatch, caplog):
        """A (scene, run) present only in one sensor is dropped with WARN."""
        import logging
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "m3nvc",
            {"cx30": ["medium", "cx30", "split_runs"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        # Audio has runs (1,6), (1,7), (2,6); seismic has (1,6), (1,7) only.
        # (2,6) should be dropped with reason="single_sensor".
        _write_split_runs_parquet(
            data_root / "train" / "m3nvc_audio_cx30_rs1.parquet",
            [(1, 6, TEST_AUDIO_W * 4), (1, 7, TEST_AUDIO_W * 4),
             (2, 6, TEST_AUDIO_W * 4)],
        )
        _write_split_runs_parquet(
            data_root / "train" / "m3nvc_seismic_cx30_rs1.parquet",
            [(1, 6, TEST_SEISMIC_W * 4), (1, 7, TEST_SEISMIC_W * 4)],
        )

        with caplog.at_level(logging.WARNING):
            SensorDataset(
                parquet_dir=data_root / "train", config=small_cfg, is_train=True,
                cache_dir=tmp_path / "raw_cache",
                use_id_split=True, role="train",
                id_root=data_root, id_cache_dir=tmp_path / "id_cache",
            )
        # WARN must mention the dropped (scene, run) and the reason
        warn_text = " ".join(r.message for r in caplog.records)
        assert "(2, 6)" in warn_text
        assert "single_sensor" in warn_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split_dataset.py -v`
Expected: FAIL — `SensorDataset.__init__()` doesn't accept the new kwargs.

- [ ] **Step 3: Update `SensorDataset.__init__` and routing logic**

In `crl-train/crl_vehicle/data/dataset.py`, modify the constructor and `_load_data` / `_build_from_parquet`. Locate the existing `__init__` (around line 118–137) and replace with:

```python
    def __init__(
        self,
        parquet_dir: str | Path,
        config: CRLConfig,
        is_train: bool = True,
        cache_dir: Path | None = None,
        *,
        use_id_split: bool = False,
        role: str = "train",
        id_root: str | Path | None = None,
        id_cache_dir: Path | None = None,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.cfg = config
        self.is_train = is_train
        self.use_id_split = use_id_split
        self.role = role
        self.id_root = Path(id_root) if id_root is not None else None
        self.id_cache_dir = Path(id_cache_dir) if id_cache_dir is not None else None
        # metadata store: (stem, seg_key) → {path, n_windows}
        self._cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        # Shared-memory tensor store: (stem, seg_key) → Tensor(N, W) in shared memory.
        # Populated once at init, shared across all DataLoader workers with zero duplication.
        self._data_cache: dict[str, dict] = {"audio": {}, "seismic": {}}
        self._index: list = []   # [(gkey, w_idx, vtype, det_label, audio_seg_id, seismic_seg_id)]
        self._groups: dict = {}  # gkey → group metadata

        if self.use_id_split:
            if self.id_root is None:
                raise ValueError("use_id_split=True requires id_root")
            if self.role not in ("train", "val", "test"):
                raise ValueError(f"role must be 'train'|'val'|'test' (got {self.role!r})")

        self._load_data(cache_dir)
```

Replace `_load_data` (around line 142–147) with:

```python
    def _load_data(self, cache_dir: Path | None) -> None:
        if self.use_id_split:
            from crl_vehicle.data.id_split import load_or_build_manifest
            self._id_manifest = load_or_build_manifest(
                id_root=self.id_root,
                mapping=DATASET_VEHICLE_MAP,
                window_sizes={
                    "audio":   self.cfg.modality_cfg("audio").window_size,
                    "seismic": self.cfg.modality_cfg("seismic").window_size,
                },
                cache_dir=self.id_cache_dir if self.id_cache_dir is not None
                           else self.id_root.parent / "id_cache",
            )
            # Scan all subdirs of id_root, dedupe by stem
            seen: dict[str, Path] = {}
            for p in sorted(self.id_root.glob("*/*.parquet")):
                seen.setdefault(p.stem, p)
            parquet_files = sorted(seen.values())
        else:
            self._id_manifest = None
            parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found "
                f"({'id_root=' + str(self.id_root) if self.use_id_split else 'parquet_dir=' + str(self.parquet_dir)})"
            )
        self._build_from_parquet(parquet_files)
        self._preload_shared(cache_dir)
```

Locate the per-window emission loop in `_build_from_parquet` (the loop currently starting `for w in range(n_windows):` near line 235) and replace with:

```python
            # ID-split routing: filter the window range based on marker / role
            if self.use_id_split:
                window_iter = self._id_split_window_indices(
                    ds=ds, vehicle=vehicle, rs=rs, n_windows=n_windows,
                )
                if window_iter is None:
                    continue  # group not present in this role
            else:
                window_iter = range(n_windows)

            for w in window_iter:
                if combined_present[w]:
                    w_vtype, w_det_label = vtype, det_label
                else:
                    w_vtype, w_det_label = LABEL_BACKGROUND, 0
                self._index.append((gkey, w, w_vtype, w_det_label, audio_seg_id, seismic_seg_id))
```

Then add a new helper method on `SensorDataset` immediately after `_build_from_parquet`:

```python
    def _id_split_window_indices(
        self, ds: str, vehicle: str, rs: str, n_windows: int,
    ) -> list[int] | None:
        """Return window indices this group contributes to self.role, or None to skip.

        - "train"/"val"/"test" markers: full range iff marker matches role.
        - "split"/"split_runs" markers: indices from manifest's
          split_assignments[role] for this group_key.
        - background / unknown: routed to "train".
        """
        ds_map = DATASET_VEHICLE_MAP.get(ds, {})
        entry = ds_map.get(vehicle)

        # Background or unknown → train only
        if entry is None or len(entry) < 3:
            if self.role == "train":
                return list(range(n_windows))
            return None

        marker = entry[2]
        if marker in ("train", "val", "test"):
            return list(range(n_windows)) if marker == self.role else None

        if marker in ("split", "split_runs"):
            gkey_str = f"{ds}__{vehicle}__{rs}"
            group = self._id_manifest.get("groups", {}).get(gkey_str)
            if group is None:
                return None
            intervals = group.get("split_assignments", {}).get(self.role, [])
            indices: list[int] = []
            for start, end in intervals:
                indices.extend(range(int(start), min(int(end), n_windows)))
            return indices if indices else None

        # Unknown marker — skip with a warning
        import logging
        logging.getLogger(__name__).warning(
            f"id_split: unknown marker {marker!r} for {ds}/{vehicle} — skipping"
        )
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
cd crl-train && .venv/bin/pytest tests/data/test_id_split_dataset.py tests/data/test_dataset.py tests/data/test_disk_cache.py -v
```
Expected: PASS for all the new tests and PASS for all existing dataset tests (ensures the default-off path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add crl-train/crl_vehicle/data/dataset.py crl-train/tests/data/test_id_split_dataset.py
git commit -m "feat(dataset): wire SensorDataset to ID split manifest"
```

---

## Task 10: `train.py` CLI flags

**Files:**
- Modify: `crl-train/train.py`

Add `--use-id-split` and `--id-root`; route them into the two `SensorDataset` constructors. When the flag is set, ignore `--data-dir`/`--val-dir` (warn at startup).

- [ ] **Step 1: Write a failing manual-style integration test**

Append to `crl-train/tests/data/test_id_split_dataset.py`:

```python
import subprocess
import sys


class TestIdSplitCli:
    def test_use_id_split_flag_parsed(self):
        # Just check the flag is recognized — running with --help should
        # print it without error.
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True, text=True, cwd="..",
        )
        # Note: cwd is set by pytest to the repo root or crl-train.
        # The flag must appear in help output:
        assert "--use-id-split" in result.stdout, result.stdout
        assert "--id-root" in result.stdout, result.stdout
```

Note: the cwd handling for subprocess is fragile. If running tests from `crl-train/` directly, use `cwd="."`. Adjust the test based on the project's `pytest.ini` / `pyproject.toml` `rootdir`. If unclear, replace this test with a direct argparse check:

```python
import importlib
import sys

class TestIdSplitCli:
    def test_use_id_split_flag_parsed(self, monkeypatch):
        # Import train.py's parser directly
        sys.path.insert(0, str(Path(__file__).parents[2]))  # crl-train/
        import train
        monkeypatch.setattr("sys.argv", ["train.py", "--use-id-split", "--id-root", "/tmp/x"])
        args = train.parse_args()
        assert args.use_id_split is True
        assert args.id_root == "/tmp/x"

    def test_use_id_split_default_false(self, monkeypatch):
        sys.path.insert(0, str(Path(__file__).parents[2]))
        import train
        monkeypatch.setattr("sys.argv", ["train.py"])
        args = train.parse_args()
        assert args.use_id_split is False
```

Use the second form — it's more deterministic.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split_dataset.py::TestIdSplitCli -v`
Expected: FAIL — `--use-id-split` is not recognized.

- [ ] **Step 3: Add the CLI flags**

In `crl-train/train.py`, in `parse_args()` (around line 23–85), add two flags **immediately after** the existing `--val-dir` line:

```python
    p.add_argument("--use-id-split", action="store_true",
                   help="Use the in-distribution split schema "
                        "(see docs/superpowers/specs/2026-04-25-id-split-schema-design.md). "
                        "When set, train/val/test assignments come from "
                        "DATASET_VEHICLE_MAP markers; --data-dir and --val-dir are ignored.")
    p.add_argument("--id-root", default="../data_files/parsed/",
                   help="Parent dir containing train/, val/, test/. "
                        "Used only when --use-id-split is set.")
```

- [ ] **Step 4: Wire the flags into dataset construction**

In `main()`, locate the two `SensorDataset(...)` calls (around lines 134–135):

```python
    train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
    val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)
```

Replace with:

```python
    if args.use_id_split:
        if args.data_dir != p.get_default("data_dir") or args.val_dir != p.get_default("val_dir"):
            print("WARNING: --use-id-split is set; --data-dir and --val-dir are ignored.")
        id_cache_dir = Path("saved_crl/id_cache")
        train_ds = SensorDataset(
            args.data_dir, cfg, is_train=True, cache_dir=cache_dir,
            use_id_split=True, role="train",
            id_root=args.id_root, id_cache_dir=id_cache_dir,
        )
        val_ds = SensorDataset(
            args.val_dir, cfg, is_train=False, cache_dir=cache_dir,
            use_id_split=True, role="val",
            id_root=args.id_root, id_cache_dir=id_cache_dir,
        )
    else:
        train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
        val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)
```

Note: `p.get_default(...)` requires `p` to be in scope. Since `p` is local to `parse_args`, replace the WARNING-guard line with a simpler check — just always emit the warning when the flag is set, without comparing to defaults:

```python
    if args.use_id_split:
        print("INFO: --use-id-split is set; --data-dir and --val-dir are ignored, "
              f"reading splits from DATASET_VEHICLE_MAP under id_root={args.id_root}")
        ...
```

- [ ] **Step 5: Run the CLI tests**

Run: `cd crl-train && .venv/bin/pytest tests/data/test_id_split_dataset.py::TestIdSplitCli -v`
Expected: PASS.

- [ ] **Step 6: End-to-end smoke (manual sanity check, no commit)**

Run a one-epoch dry run to make sure imports and dataset construction work end-to-end:

```bash
cd crl-train && .venv/bin/python train.py --use-id-split --crl-epochs 1 --steps-per-epoch 2 --num-workers 0 --batch-size 4
```

Expected: Training starts, prints class weights, runs ≤2 batches, exits cleanly. If it errors before the first batch, debug — most likely a path / parquet issue in `id_root`.

- [ ] **Step 7: Commit**

```bash
git add crl-train/train.py crl-train/tests/data/test_id_split_dataset.py
git commit -m "feat(train): add --use-id-split and --id-root CLI flags"
```

---

## Task 11: Full regression — run the entire test suite

- [ ] **Step 1: Run the full test suite**

Run: `cd crl-train && .venv/bin/pytest -v`
Expected: PASS for all tests, including the previously-existing ones (no regressions).

If any pre-existing test fails, investigate before assuming the change is at fault — check `git stash && pytest && git stash pop` to confirm it was passing before.

- [ ] **Step 2: Commit (if any test infrastructure tweaks were needed)**

If no changes needed, skip. Otherwise:

```bash
git add -A
git commit -m "test: stabilize regression suite for id-split changes"
```

---

## Self-Review Notes (already applied)

- Spec coverage: every section of the spec is mapped to a task — `"split"` (Task 2), `"split_runs"` (Tasks 3–5), pairing rule (Task 4), per-`(scene,run)` independence (Task 4), manifest hash + cache (Tasks 6, 8), `SensorDataset` API (Task 9), `train.py` CLI (Task 10), tests for all of the above.
- Type consistency: `extract_runs` returns `dict[(scene,run), (start_w, end_w)]` consumed unchanged by `pair_runs`; `pair_runs` output consumed unchanged by `partition_runs_50_25_25`; `build_manifest` returns the manifest dict consumed by `_id_split_window_indices`.
- Placeholder scan: no `TBD`/`TODO`/"add error handling"; every code step shows the actual code. The Task 10 cwd-handling alternative is provided inline so the engineer doesn't have to invent one.
- The existing tests (`test_dataset.py`, `test_disk_cache.py`) act as the OOD-path identity check (Task 9, Step 4), making a separate "golden index" test redundant.
