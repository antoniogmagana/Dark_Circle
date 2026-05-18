"""Standalone tests for crl_vehicle.data._table_io.

These tests verify that the format dispatcher produces identical results
for parquet and CSV inputs derived from the same source data, that the
priority rule prefers parquet, and that the row-count cache short-circuits
CSV scans on repeated calls.

No dataset.py wiring is exercised here — that comes in the next commit.
"""

from __future__ import annotations

import csv as _csv
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from crl_vehicle.data import _table_io

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, amplitude: np.ndarray, present: np.ndarray) -> None:
    table = pa.table({"amplitude": amplitude.astype(np.float32), "present": present.astype(bool)})
    pq.write_table(table, path)


def _write_csv(
    path: Path,
    amplitude: np.ndarray,
    present: np.ndarray,
    *,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a CSV with `amplitude`, `present`, and optionally extra columns."""
    cols = {"amplitude": amplitude, "present": present}
    if extra_cols:
        cols.update(extra_cols)
    header = list(cols.keys())
    rows = list(zip(*cols.values(), strict=True))
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _signal_and_labels(n_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    amp = rng.standard_normal(n_samples).astype(np.float32)
    # Per-sample present pattern: first half True, second half False.
    present = np.zeros(n_samples, dtype=bool)
    present[: n_samples // 2] = True
    return amp, present


# ---------------------------------------------------------------------------
# read_amplitude
# ---------------------------------------------------------------------------


def test_read_amplitude_parquet_matches_csv(tmp_path: Path) -> None:
    """The two formats should produce bit-identical windows when contents match."""
    n = 16000 * 3  # exactly 3 windows at source_rate=16000
    amp, present = _signal_and_labels(n)

    pq_path = tmp_path / "x.parquet"
    csv_path = tmp_path / "x.csv"
    _write_parquet(pq_path, amp, present)
    _write_csv(csv_path, amp, present)

    target_window = 16000
    src_sr = 16000
    tgt_sr = 16000

    out_pq = _table_io.read_amplitude(pq_path, target_window, src_sr, tgt_sr)
    out_csv = _table_io.read_amplitude(csv_path, target_window, src_sr, tgt_sr)

    assert out_pq.shape == (3, target_window)
    assert out_csv.shape == out_pq.shape
    np.testing.assert_array_equal(out_pq, out_csv)


def test_read_amplitude_resamples_csv(tmp_path: Path) -> None:
    """CSV path must apply the same resampling as the parquet path."""
    n = 1600 * 2  # 2 seconds at source_rate=1600 (m3nvc audio)
    amp, present = _signal_and_labels(n)

    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, amp, present)

    out = _table_io.read_amplitude(csv_path, target_window_size=16000, source_rate=1600, target_rate=16000)
    # After resampling 2 seconds @ 1600Hz to 16000Hz we expect 2 windows of 16000.
    assert out.shape == (2, 16000)
    assert out.dtype == np.float32


def test_read_amplitude_rejects_unknown_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "x.parq"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file type"):
        _table_io.read_amplitude(bad, 16000, 16000, 16000)


# ---------------------------------------------------------------------------
# read_present
# ---------------------------------------------------------------------------


def test_read_present_parquet_matches_csv(tmp_path: Path) -> None:
    n = 16000 * 4
    amp, present = _signal_and_labels(n)
    # Force exactly 2 windows present, 2 not, with clean 50% boundary at the
    # window midpoint to land on the strict > 0.5 majority threshold.
    present = np.zeros(n, dtype=bool)
    present[: 16000 * 2 + 1] = True  # window 0,1 majority True; window 2 just under

    pq_path = tmp_path / "x.parquet"
    csv_path = tmp_path / "x.csv"
    _write_parquet(pq_path, amp, present)
    _write_csv(csv_path, amp, present)

    pq_out = _table_io.read_present(pq_path, source_rate=16000)
    csv_out = _table_io.read_present(csv_path, source_rate=16000)

    assert pq_out.shape == (4,)
    np.testing.assert_array_equal(pq_out, csv_out)
    # Sanity: windows 0,1 majority True; 2 (one True sample) and 3 (none) False.
    assert pq_out.tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# read_scene_run
# ---------------------------------------------------------------------------


def test_read_scene_run_parquet_and_csv_match(tmp_path: Path) -> None:
    n = 100
    amp, present = _signal_and_labels(n)
    scene = np.arange(n, dtype=np.int64) // 25  # 4 scenes
    run = np.arange(n, dtype=np.int64) // 50  # 2 runs

    pq_path = tmp_path / "x.parquet"
    table = pa.table(
        {
            "amplitude": amp,
            "present": present,
            "scene_id": scene,
            "run_id": run,
        }
    )
    pq.write_table(table, pq_path)

    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, amp, present, extra_cols={"scene_id": scene, "run_id": run})

    s_pq, r_pq = _table_io.read_scene_run(pq_path)
    s_csv, r_csv = _table_io.read_scene_run(csv_path)

    np.testing.assert_array_equal(s_pq, s_csv)
    np.testing.assert_array_equal(r_pq, r_csv)


def test_read_scene_run_missing_column_raises_parquet(tmp_path: Path) -> None:
    n = 10
    amp, present = _signal_and_labels(n)
    pq_path = tmp_path / "x.parquet"
    _write_parquet(pq_path, amp, present)
    with pytest.raises(ValueError, match="missing required column"):
        _table_io.read_scene_run(pq_path)


def test_read_scene_run_missing_column_raises_csv(tmp_path: Path) -> None:
    n = 10
    amp, present = _signal_and_labels(n)
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, amp, present)
    with pytest.raises(ValueError, match="missing required column"):
        _table_io.read_scene_run(csv_path)


# ---------------------------------------------------------------------------
# get_num_rows
# ---------------------------------------------------------------------------


def test_get_num_rows_parquet_is_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet path must not touch the data columns — only the footer."""
    n = 50_000
    amp, present = _signal_and_labels(n)
    pq_path = tmp_path / "x.parquet"
    _write_parquet(pq_path, amp, present)

    # If pq.read_table is called on the parquet path for row count, fail.
    def _no_read_table(*args, **kwargs):  # pragma: no cover
        raise AssertionError("pq.read_table should not be called for parquet row counts")

    monkeypatch.setattr(_table_io.pq, "read_table", _no_read_table)
    assert _table_io.get_num_rows(pq_path) == n


def test_get_num_rows_csv_uses_cache(tmp_path: Path) -> None:
    """Second call with the same cache must not re-scan the CSV."""
    n = 10_000
    amp, present = _signal_and_labels(n)
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, amp, present)

    cache: dict[str, int] = {}
    assert _table_io.get_num_rows(csv_path, row_count_cache=cache) == n
    assert cache == {csv_path.as_posix(): n}

    # Sabotage the file; cached read must still return the original count.
    csv_path.write_text("amplitude,present\n0.0,True\n")
    assert _table_io.get_num_rows(csv_path, row_count_cache=cache) == n


def test_get_num_rows_csv_without_cache_rescans(tmp_path: Path) -> None:
    """No cache → recompute from disk every call."""
    n = 200
    amp, present = _signal_and_labels(n)
    csv_path = tmp_path / "x.csv"
    _write_csv(csv_path, amp, present)

    assert _table_io.get_num_rows(csv_path) == n
    # Rewrite shorter; uncached read must reflect new length.
    _write_csv(csv_path, amp[:50], present[:50])
    assert _table_io.get_num_rows(csv_path) == 50


def test_get_num_rows_csv_falls_back_to_amplitude(tmp_path: Path) -> None:
    """A CSV without `present` should still yield a row count via amplitude."""
    csv_path = tmp_path / "x.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["amplitude"])
        for v in range(123):
            w.writerow([float(v)])
    assert _table_io.get_num_rows(csv_path) == 123


def test_get_num_rows_rejects_unknown_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file type"):
        _table_io.get_num_rows(bad)


# ---------------------------------------------------------------------------
# merge_with_parquet_priority
# ---------------------------------------------------------------------------


def test_merge_prefers_parquet_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pq_a = tmp_path / "a.parquet"
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    for p in (pq_a, csv_a, csv_b):
        p.write_bytes(b"")

    logger = logging.getLogger("test_merge")
    with caplog.at_level(logging.WARNING, logger="test_merge"):
        result = _table_io.merge_with_parquet_priority([pq_a], [csv_a, csv_b], logger=logger)

    # a: parquet wins; csv_a shadowed. b: csv kept.
    assert pq_a in result
    assert csv_b in result
    assert csv_a not in result
    assert len(result) == 2
    assert any("a.parquet and a.csv" in rec.message for rec in caplog.records)


def test_merge_no_conflicts_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pq_a = tmp_path / "a.parquet"
    csv_b = tmp_path / "b.csv"
    for p in (pq_a, csv_b):
        p.write_bytes(b"")

    with caplog.at_level(logging.WARNING):
        result = _table_io.merge_with_parquet_priority([pq_a], [csv_b])

    assert result == sorted([pq_a, csv_b])
    assert not any("ignoring CSV" in rec.message for rec in caplog.records)


def test_merge_output_is_sorted(tmp_path: Path) -> None:
    paths_pq = [tmp_path / f"{name}.parquet" for name in ("zeta", "alpha")]
    paths_csv = [tmp_path / f"{name}.csv" for name in ("mike", "bravo")]
    for p in [*paths_pq, *paths_csv]:
        p.write_bytes(b"")
    result = _table_io.merge_with_parquet_priority(paths_pq, paths_csv)
    assert result == sorted(result)
