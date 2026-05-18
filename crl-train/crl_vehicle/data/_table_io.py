"""Format-agnostic table reads for parquet and CSV sources.

The training pipeline historically read pyarrow Parquet exclusively. This
module is the single dispatch point that lets the rest of the data layer
read the same logical columns from either `.parquet` or `.csv` without
caring which is on disk.

Parquet paths call straight through to pyarrow.parquet (zero behavior change).
CSV paths call pyarrow.csv with `include_columns` to keep parsing scoped to
the column actually needed. CSV is meaningfully slower than parquet — see
`get_num_rows` for the row-count caching contract that compensates for it.

Conflict rule: when the caller hands us a glob result with both a `foo.parquet`
and `foo.csv` for the same stem, prefer parquet and warn — see
`merge_with_parquet_priority`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)

# Suffixes we route. Anything else is rejected at the dispatch boundary so a
# typo (e.g. `.parq`) fails loudly rather than silently dropping a file.
_PARQUET_SUFFIXES = {".parquet"}
_CSV_SUFFIXES = {".csv"}


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def _is_parquet(path: Path) -> bool:
    return _suffix(path) in _PARQUET_SUFFIXES


def _is_csv(path: Path) -> bool:
    return _suffix(path) in _CSV_SUFFIXES


def _read_csv_column(path: Path, column: str) -> pa.Table:
    """Read a single column from a CSV via pyarrow, with column-scoped parsing."""
    convert_opts = pa_csv.ConvertOptions(include_columns=[column])
    return pa_csv.read_csv(path, convert_options=convert_opts)


def _resample_to_target(arr: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Local copy of dataset._resample_to_target to avoid an import cycle."""
    if source_rate == target_rate:
        return arr
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(arr).float().unsqueeze(0)
    out = AF.resample(t, orig_freq=source_rate, new_freq=target_rate)
    return out.squeeze(0).numpy().astype(np.float32, copy=False)


def read_amplitude(
    path: Path,
    target_window_size: int,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Return float32 (N_windows, target_window_size) from `amplitude` column.

    Matches dataset._read_parquet_numpy exactly for parquet paths.
    """
    path = Path(path)
    if _is_parquet(path):
        table = pq.read_table(path, columns=["amplitude"], use_threads=True)
    elif _is_csv(path):
        table = _read_csv_column(path, "amplitude")
    else:
        raise ValueError(f"Unsupported file type for amplitude read: {path}")

    arr = table.column("amplitude").to_numpy().astype(np.float32)
    arr = _resample_to_target(arr, source_rate, target_rate)
    n_windows = len(arr) // target_window_size
    return arr[: n_windows * target_window_size].reshape(n_windows, target_window_size)


def read_present(path: Path, source_rate: int) -> np.ndarray:
    """Return bool (N_windows,) per-window majority vote on `present`.

    Matches dataset._read_parquet_present exactly for parquet paths. Window
    boundary is `source_rate` samples regardless of any downstream resampling.
    """
    path = Path(path)
    if _is_parquet(path):
        table = pq.read_table(path, columns=["present"], use_threads=True)
    elif _is_csv(path):
        table = _read_csv_column(path, "present")
    else:
        raise ValueError(f"Unsupported file type for present read: {path}")

    arr = table.column("present").to_numpy()
    n_windows = len(arr) // source_rate
    arr = arr[: n_windows * source_rate].reshape(n_windows, source_rate)
    return arr.mean(axis=1) > 0.5


def read_scene_run(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (scene_id, run_id) int arrays. Raises ValueError if columns missing.

    Used by id_split.extract_runs for the `split_runs` marker.
    """
    path = Path(path)
    if _is_parquet(path):
        schema_cols = set(pq.read_schema(path).names)
        for col in ("scene_id", "run_id"):
            if col not in schema_cols:
                raise ValueError(
                    f"{path.name}: missing required column {col!r} for split_runs marker"
                )
        table = pq.read_table(path, columns=["scene_id", "run_id"])
    elif _is_csv(path):
        convert_opts = pa_csv.ConvertOptions(include_columns=["scene_id", "run_id"])
        try:
            table = pa_csv.read_csv(path, convert_options=convert_opts)
        except KeyError as e:
            raise ValueError(
                f"{path.name}: missing required column for split_runs marker ({e})"
            ) from e
        present_cols = set(table.column_names)
        for col in ("scene_id", "run_id"):
            if col not in present_cols:
                raise ValueError(
                    f"{path.name}: missing required column {col!r} for split_runs marker"
                )
    else:
        raise ValueError(f"Unsupported file type for scene/run read: {path}")

    return table.column("scene_id").to_numpy(), table.column("run_id").to_numpy()


def get_num_rows(
    path: Path,
    *,
    row_count_cache: dict[str, int] | None = None,
) -> int:
    """Return total row count.

    Parquet: O(1) via footer metadata, regardless of cache.
    CSV: cache hit is O(1); cache miss scans the file once via the smallest
    column (`present`) to minimize parse cost. Pass the same dict on subsequent
    calls within a build to avoid re-scanning.

    Cache key is the path's POSIX string so it survives Path equality quirks.
    """
    path = Path(path)
    if _is_parquet(path):
        return pq.read_metadata(path).num_rows

    if not _is_csv(path):
        raise ValueError(f"Unsupported file type for row count: {path}")

    cache_key = path.as_posix()
    if row_count_cache is not None and cache_key in row_count_cache:
        return row_count_cache[cache_key]

    # Cache miss: scan once via the smallest column. `present` (bool) is
    # cheaper to parse than amplitude (float). If the file has no `present`,
    # fall back to amplitude.
    try:
        table = _read_csv_column(path, "present")
    except (pa.ArrowInvalid, KeyError):
        table = _read_csv_column(path, "amplitude")
    n = table.num_rows

    if row_count_cache is not None:
        row_count_cache[cache_key] = n
    return n


def merge_with_parquet_priority(
    parquet_files: list[Path],
    csv_files: list[Path],
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Combine parquet and CSV source lists, preferring parquet on stem conflicts.

    A CSV is included only if no parquet with the same stem exists. Shadowed
    CSVs emit a single warning per conflict via the supplied (or module) logger.
    Output is sorted by path for deterministic ordering downstream.
    """
    log = logger if logger is not None else _LOG
    parquet_stems = {p.stem for p in parquet_files}
    kept_csv: list[Path] = []
    for c in csv_files:
        if c.stem in parquet_stems:
            log.warning(
                "Both %s.parquet and %s.csv found; using parquet, ignoring CSV.",
                c.stem,
                c.stem,
            )
            continue
        kept_csv.append(c)
    return sorted([*parquet_files, *kept_csv])
