"""Manifest builder for the ID split schema (--use-id-split).

Computes per-group window-range manifests from DATASET_VEHICLE_MAP
markers ("split", "split_runs"). Designed to be pure (no torch, no
CUDA) and cache-friendly.

See docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


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
    # Check schema before reading to raise a clear ValueError on missing columns.
    schema_cols = set(pq.read_schema(parquet_path).names)
    for col in ("scene_id", "run_id"):
        if col not in schema_cols:
            raise ValueError(
                f"{parquet_path.name}: missing required column {col!r} "
                f"for split_runs marker"
            )
    table = pq.read_table(parquet_path, columns=["scene_id", "run_id"])
    cols = set(table.column_names)

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
