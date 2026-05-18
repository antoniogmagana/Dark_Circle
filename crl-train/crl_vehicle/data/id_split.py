"""Manifest builder for the ID split schema (--use-id-split).

Computes per-group window-range manifests from DATASET_VEHICLE_MAP
markers ("split", "split_runs"). Designed to be pure (no torch, no
CUDA) and cache-friendly.

See docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from crl_vehicle.data import _table_io

logger = logging.getLogger(__name__)

_KNOWN_DATASETS = {"focal", "iobt", "m3nvc"}

# Per-dataset source sample rates. Mirror of crl_vehicle/data/dataset.py
# (kept here to avoid a cross-module import cycle). The id-split manifest
# always thinks in 1-second windows, so window math uses *source* rate
# (the on-disk samples-per-second), which makes audio and seismic ranges
# directly comparable in seconds regardless of post-load resampling.
_SOURCE_RATES = {
    "focal": {"audio": 16000, "seismic": 100},
    "iobt": {"audio": 16000, "seismic": 100},
    "m3nvc": {"audio": 1600, "seismic": 200},
}


def _source_rate_for_stem(stem: str, sensor: str) -> int:
    dataset = stem.split("_", 1)[0]
    rates = _SOURCE_RATES.get(dataset)
    if rates is None:
        raise ValueError(
            f"Unknown dataset prefix in stem {stem!r}; " f"expected one of {sorted(_SOURCE_RATES)}"
        )
    return rates[sensor]


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
        "val": [(0, half)],
        "test": [(half, n_paired)],
    }


def extract_runs(
    source_path: Path,
    window_size: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Extract per-run window ranges from a parquet or CSV with scene_id/run_id.

    Args:
        source_path: .parquet or .csv with columns 'scene_id' (int64), 'run_id' (int64).
        window_size: source samples per 1-second window (i.e. the source
            sample rate of the file on disk — 16000 for focal/iobt audio,
            1600 for m3nvc audio, 100 for focal/iobt seismic, 200 for m3nvc
            seismic). Window indices are computed in 1-second units so audio
            and seismic ranges become directly comparable.

    Returns:
        {(scene_id, run_id): (start_window, end_window)} where ranges are
        half-open [start, end) and use ceil(start_sample / W),
        floor(end_sample / W) to drop windows that straddle a run boundary.

    Raises:
        ValueError: if scene_id or run_id columns are missing, or if any
            (scene, run) key is non-contiguous in the file.
    """
    source_path = Path(source_path)
    scene, run = _table_io.read_scene_run(source_path)
    n = len(scene)
    if n == 0:
        return {}

    # Find block boundaries: positions where (scene, run) changes.
    key_changed = (scene[1:] != scene[:-1]) | (run[1:] != run[:-1])
    boundaries = np.flatnonzero(key_changed) + 1  # start indices of new blocks
    block_starts = np.concatenate(([0], boundaries))
    block_ends = np.concatenate((boundaries, [n]))

    # Verify contiguity: each (scene, run) appears in exactly one block
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for s, e in zip(block_starts, block_ends, strict=False):
        key = (int(scene[s]), int(run[s]))
        if key in seen:
            raise ValueError(
                f"{source_path.name}: (scene_id={key[0]}, run_id={key[1]}) "
                f"appears in non-contiguous blocks (sample ranges "
                f"[{seen[key][0]},{seen[key][1]}) and [{s},{e}))"
            )
        seen[key] = (int(s), int(e))

    # Convert sample ranges to window ranges (1 window = 1 second of source).
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (s, e) in seen.items():
        w_start = math.ceil(s / window_size)
        w_end = e // window_size  # floor of e
        out[key] = (w_start, w_end)
    return out


def pair_runs(
    audio_runs: dict[tuple[int, int], tuple[int, int]],
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
        end = min(a[1], s[1])
        if start >= end:
            dropped.append({"run_key": key, "reason": "empty_intersection"})
            continue
        paired[key] = (start, end)
    return paired, dropped


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
        deficits = {b: _TARGET_RATIOS[b] * total - bucket_totals[b] for b in _TIE_ORDER}
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


def compute_manifest_hash(
    mapping: dict,
    window_sizes: dict[str, int],
    source_files: list[tuple[str, Path]],
) -> str:
    """SHA-256 over (mapping, window sizes, sorted source-file mtimes_ns).

    Args:
        mapping: DATASET_VEHICLE_MAP (or a subset).
        window_sizes: {"audio": int, "seismic": int}.
        source_files: list of (stem, path) for files (parquet or CSV) whose
            routing depends on per-file computation (split / split_runs).

    Returns:
        Hex SHA-256 digest (64 chars).
    """
    # Sort source files by stem for order invariance
    sources_sorted = sorted(source_files, key=lambda kv: kv[0])
    mtimes = [(stem, Path(p).stat().st_mtime_ns) for stem, p in sources_sorted]

    payload = {
        "mapping": mapping,
        "window_sizes": window_sizes,
        "sources": mtimes,
        "source_rates": _SOURCE_RATES,  # invalidates cache when rates change
        "schema_version": 2,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def _file_n_windows(
    path: Path,
    window_size: int,
    *,
    row_count_cache: dict[str, int] | None = None,
) -> int:
    """Number of complete 1-second windows in the source file.

    `window_size` is samples-per-second at the file's source rate
    (e.g. 16000 for focal/iobt audio, 100 for focal/iobt seismic).

    Parquet reads are O(1) via footer metadata. CSV reads are cached via
    `row_count_cache` (shared across the build) to avoid re-scanning.
    """
    return _table_io.get_num_rows(path, row_count_cache=row_count_cache) // window_size


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
        window_sizes: kept for backward-compat; IGNORED. Per-file source
            rates are looked up from the dataset prefix via _SOURCE_RATES.
            Pass any value (e.g. {"audio": 16000, "seismic": 200}); it
            still flows into the manifest hash so caches invalidate when
            the caller's notion of canonical sizes changes.

    Returns:
        Manifest dict matching the schema in the design spec.
    """
    id_root = Path(id_root)
    # Scan all subdirs for both .parquet and .csv. Within each stem, prefer
    # parquet via _table_io.merge_with_parquet_priority.
    pq_by_stem: dict[str, Path] = {}
    csv_by_stem: dict[str, Path] = {}
    for p in id_root.glob("*/*.parquet"):
        pq_by_stem.setdefault(p.stem, p)
    for p in id_root.glob("*/*.csv"):
        csv_by_stem.setdefault(p.stem, p)
    merged = _table_io.merge_with_parquet_priority(
        list(pq_by_stem.values()), list(csv_by_stem.values()), logger=logger
    )
    all_files: dict[str, Path] = {p.stem: p for p in merged}

    # Group by (dataset, vehicle, rs)
    audio_files: dict[tuple[str, str, str], Path] = {}
    seismic_files: dict[tuple[str, str, str], Path] = {}
    for stem, path in all_files.items():
        for sensor, dest in (("audio", audio_files), ("seismic", seismic_files)):
            parsed = _parse_stem(stem, sensor)
            if parsed is not None:
                dest[parsed] = path
                break

    groups: dict[str, dict] = {}
    all_keys = set(audio_files) | set(seismic_files)
    # Per-build CSV row-count cache; parquet reads are O(1) and ignore it.
    row_count_cache: dict[str, int] = {}

    for ds, vehicle, rs in sorted(all_keys):
        ds_map = mapping.get(ds, {})
        entry = ds_map.get(vehicle)
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
            audio_nw = (
                _file_n_windows(
                    a_path,
                    _source_rate_for_stem(a_path.stem, "audio"),
                    row_count_cache=row_count_cache,
                )
                if a_path
                else 0
            )
            seismic_nw = (
                _file_n_windows(
                    s_path,
                    _source_rate_for_stem(s_path.stem, "seismic"),
                    row_count_cache=row_count_cache,
                )
                if s_path
                else 0
            )
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
                "dataset": ds,
                "vehicle": vehicle,
                "rs_node": rs,
                "marker": marker,
                "split_assignments": {k: [list(iv) for iv in v] for k, v in intervals.items()},
            }

        elif marker == "split_runs":
            if a_path is None or s_path is None:
                logger.info(
                    f"id_split: skipping group {gkey!r} — split_runs requires "
                    f"both audio and seismic (have audio={a_path is not None}, "
                    f"seismic={s_path is not None})"
                )
                continue
            audio_runs = extract_runs(a_path, _source_rate_for_stem(a_path.stem, "audio"))
            seismic_runs = extract_runs(s_path, _source_rate_for_stem(s_path.stem, "seismic"))
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
                "train": [],
                "val": [],
                "test": [],
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
                "dataset": ds,
                "vehicle": vehicle,
                "rs_node": rs,
                "marker": marker,
                "split_assignments": split_assignments,
                "run_meta": run_meta,
                "dropped_runs": [
                    {
                        "run_key": f"{d['run_key'][0]}_{d['run_key'][1]}",
                        "reason": d["reason"],
                    }
                    for d in dropped
                ],
            }

    return {
        "schema_version": 2,  # bumped: window math now uses source rates per stem
        "created_unix": int(time.time()),
        "config_window_sizes": dict(window_sizes),
        "source_rates": _SOURCE_RATES,
        "groups": groups,
    }


def _collect_split_source_files(id_root: Path, mapping: dict) -> list[tuple[str, Path]]:
    """List (stem, path) for every source file whose marker is split / split_runs.

    Walks both .parquet and .csv under id_root/*/, preferring parquet on stem
    conflicts (so the manifest hash sees the same source each time and CSV
    shadowed by a parquet doesn't leak into invalidation tracking).
    """
    id_root = Path(id_root)
    pq_by_stem: dict[str, Path] = {}
    csv_by_stem: dict[str, Path] = {}
    for p in id_root.glob("*/*.parquet"):
        pq_by_stem.setdefault(p.stem, p)
    for p in id_root.glob("*/*.csv"):
        csv_by_stem.setdefault(p.stem, p)
    merged = _table_io.merge_with_parquet_priority(
        list(pq_by_stem.values()), list(csv_by_stem.values()), logger=logger
    )

    out: list[tuple[str, Path]] = []
    seen_stems: set[str] = set()
    for source in merged:
        stem = source.stem
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
                out.append((stem, source))
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
