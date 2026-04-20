# tests/data/test_disk_cache.py
"""Tests for parquet helpers and SensorDataset disk-cache logic."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from crl_vehicle.data.dataset import _read_parquet_numpy, _read_parquet_present


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet(path: Path, amplitude: np.ndarray, present: np.ndarray) -> None:
    """Write a parquet file with 'amplitude' (float32) and 'present' (bool) columns."""
    df = pd.DataFrame({
        "amplitude": amplitude.astype("float32"),
        "present":   present.astype(bool),
    })
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# _read_parquet_numpy
# ---------------------------------------------------------------------------

def test_read_parquet_numpy_shape(tmp_path):
    p = tmp_path / "test.parquet"
    n_windows, window_size = 5, 8
    amp = np.arange(n_windows * window_size, dtype="float32")
    _write_parquet(p, amp, np.ones(len(amp), dtype=bool))
    arr = _read_parquet_numpy(p, window_size)
    assert arr.shape == (n_windows, window_size)
    assert arr.dtype == np.float32


def test_read_parquet_numpy_truncates_remainder(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    amp = np.arange(11, dtype="float32")  # 2 full windows + 3 leftover
    _write_parquet(p, amp, np.ones(len(amp), dtype=bool))
    arr = _read_parquet_numpy(p, window_size)
    assert arr.shape == (2, window_size)


def test_read_parquet_numpy_values(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    amp = np.arange(8, dtype="float32")
    _write_parquet(p, amp, np.ones(len(amp), dtype=bool))
    arr = _read_parquet_numpy(p, window_size)
    np.testing.assert_array_equal(arr[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(arr[1], [4, 5, 6, 7])


# ---------------------------------------------------------------------------
# _read_parquet_present
# ---------------------------------------------------------------------------

def test_read_parquet_present_shape(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    amp = np.zeros(12, dtype="float32")
    pres = np.ones(12, dtype=bool)
    _write_parquet(p, amp, pres)
    arr = _read_parquet_present(p, window_size)
    assert arr.shape == (3,)
    assert arr.dtype == bool


def test_read_parquet_present_all_true(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    _write_parquet(p, np.zeros(8, dtype="float32"), np.ones(8, dtype=bool))
    arr = _read_parquet_present(p, window_size)
    assert arr.all()


def test_read_parquet_present_all_false(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    _write_parquet(p, np.zeros(8, dtype="float32"), np.zeros(8, dtype=bool))
    arr = _read_parquet_present(p, window_size)
    assert not arr.any()


def test_read_parquet_present_majority_strict(tmp_path):
    """Exactly 50% True → False (strict >0.5 rule)."""
    p = tmp_path / "test.parquet"
    window_size = 4
    # Window 0: [T, T, F, F] → mean=0.5, not >0.5 → False
    # Window 1: [T, T, T, F] → mean=0.75, >0.5 → True
    pres = np.array([True, True, False, False, True, True, True, False])
    _write_parquet(p, np.zeros(8, dtype="float32"), pres)
    arr = _read_parquet_present(p, window_size)
    assert arr.shape == (2,)
    assert arr[0] == False
    assert arr[1] == True


def test_read_parquet_present_truncates_remainder(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 4
    pres = np.ones(11, dtype=bool)  # 2 full windows + 3 leftover
    _write_parquet(p, np.zeros(11, dtype="float32"), pres)
    arr = _read_parquet_present(p, window_size)
    assert arr.shape == (2,)


# ---------------------------------------------------------------------------
# _read_parquet_numpy and _read_parquet_present have identical n_windows
# ---------------------------------------------------------------------------

def test_numpy_and_present_window_counts_match(tmp_path):
    p = tmp_path / "test.parquet"
    window_size = 5
    n_samples = 23  # 4 complete windows, 3 leftover
    _write_parquet(p, np.zeros(n_samples, dtype="float32"), np.ones(n_samples, dtype=bool))
    amp_arr  = _read_parquet_numpy(p, window_size)
    pres_arr = _read_parquet_present(p, window_size)
    assert amp_arr.shape[0] == pres_arr.shape[0]


# ---------------------------------------------------------------------------
# _preload_shared: old bare-tensor cache is auto-detected and regenerated
# ---------------------------------------------------------------------------

def test_old_format_cache_regenerated(tmp_path):
    """A bare Tensor .pt written by the old code is replaced with a dict on load."""
    import crl_vehicle.data.dataset as ds_mod
    from crl_vehicle.config import CRLConfig

    cfg = CRLConfig()
    W_a = cfg.modality_cfg("audio").window_size
    n_windows = 3

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    cache_dir   = tmp_path / "cache"
    cache_dir.mkdir()

    amp  = np.zeros(n_windows * W_a, dtype="float32")
    pres = np.ones(n_windows * W_a, dtype=bool)
    pq_path = parquet_dir / "focal_audio_mustang_rs1.parquet"
    _write_parquet(pq_path, amp, pres)

    # Write old-format bare tensor cache (simulate pre-fix cache file)
    stem    = "focal_audio_mustang_rs1"
    pt_path = cache_dir / f"{stem}.pt"
    old_tensor = torch.zeros(n_windows, W_a)
    torch.save(old_tensor, pt_path)
    # Make it appear newer than the parquet so mtime check passes
    import os
    os.utime(pt_path, (pq_path.stat().st_mtime + 10,) * 2)

    ds = ds_mod.SensorDataset(str(parquet_dir), cfg, cache_dir=cache_dir)

    # The cache should have been regenerated as a dict
    new_loaded = torch.load(pt_path, weights_only=True)
    assert isinstance(new_loaded, dict), "old bare-tensor cache was not regenerated as dict"
    assert "amplitude" in new_loaded
    assert "present"   in new_loaded


def test_new_format_cache_loaded(tmp_path):
    """A dict .pt written by the new code is loaded without re-reading parquet."""
    import crl_vehicle.data.dataset as ds_mod
    from crl_vehicle.config import CRLConfig

    cfg = CRLConfig()
    W_a = cfg.modality_cfg("audio").window_size
    n_windows = 2

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    cache_dir   = tmp_path / "cache"
    cache_dir.mkdir()

    amp  = np.zeros(n_windows * W_a, dtype="float32")
    pres = np.ones(n_windows * W_a, dtype=bool)
    pq_path = parquet_dir / "focal_audio_mustang_rs1.parquet"
    _write_parquet(pq_path, amp, pres)

    # First instantiation writes the new-format cache
    ds_mod.SensorDataset(str(parquet_dir), cfg, cache_dir=cache_dir)

    pt_path = cache_dir / "focal_audio_mustang_rs1.pt"
    assert pt_path.exists()
    loaded = torch.load(pt_path, weights_only=True)
    assert isinstance(loaded, dict)
    assert loaded["amplitude"].shape == (n_windows, W_a)
    assert loaded["present"].shape   == (n_windows,)
    assert loaded["present"].dtype   == torch.bool
