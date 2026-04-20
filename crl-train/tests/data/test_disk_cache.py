# tests/data/test_disk_cache.py
"""Tests for SensorDataset parquet loading (no disk cache)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset, _read_parquet_numpy


def _write_dummy_parquet(path: Path, n_rows: int = 4, n_cols: int = 10) -> None:
    df = pd.DataFrame(np.random.rand(n_rows, n_cols).astype("float32"))
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# _read_parquet_numpy
# ---------------------------------------------------------------------------

def test_read_parquet_numpy_shape(tmp_path):
    p = tmp_path / "test.parquet"
    _write_dummy_parquet(p, n_rows=5, n_cols=8)
    arr = _read_parquet_numpy(p)
    assert arr.shape == (5, 8)
    assert arr.dtype == np.float32


def test_read_parquet_numpy_values(tmp_path):
    p = tmp_path / "test.parquet"
    data = np.arange(12, dtype="float32").reshape(3, 4)
    pd.DataFrame(data).to_parquet(p)
    arr = _read_parquet_numpy(p)
    np.testing.assert_array_equal(arr, data)


def test_read_parquet_numpy_drops_string_cols(tmp_path):
    p = tmp_path / "mixed.parquet"
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3.0, 4.0]})
    df.to_parquet(p)
    arr = _read_parquet_numpy(p)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# SensorDataset raises on empty directory
# ---------------------------------------------------------------------------

def test_sensor_dataset_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        SensorDataset(str(tmp_path), CRLConfig())


# ---------------------------------------------------------------------------
# SensorDataset loads parquet without cache_dir
# ---------------------------------------------------------------------------

def test_sensor_dataset_loads_without_cache(tmp_path, monkeypatch):
    import crl_vehicle.data.dataset as ds_mod

    built = []

    def fake_build(self, files):
        self._cache = {
            "audio":   {("stem_a", None): {"data": np.zeros((2, 16000), dtype="float32"), "n_windows": 2}},
            "seismic": {("stem_s", None): {"data": np.zeros((2, 200),   dtype="float32"), "n_windows": 2}},
        }
        self._index  = []
        self._groups = {}
        built.append(True)

    monkeypatch.setattr(ds_mod.SensorDataset, "_build_from_parquet", fake_build)

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_dummy_parquet(parquet_dir / "iobt_audio_polaris0150pm_rs0.parquet")

    ds_mod.SensorDataset(str(parquet_dir), CRLConfig())
    assert len(built) == 1


def test_sensor_dataset_cache_dir_ignored(tmp_path, monkeypatch):
    """Passing cache_dir no longer affects behaviour — build always runs."""
    import crl_vehicle.data.dataset as ds_mod

    built = []

    def fake_build(self, files):
        self._cache = {
            "audio":   {("stem_a", None): {"data": np.zeros((1, 16000), dtype="float32"), "n_windows": 1}},
            "seismic": {("stem_s", None): {"data": np.zeros((1, 200),   dtype="float32"), "n_windows": 1}},
        }
        self._index  = []
        self._groups = {}
        built.append(True)

    monkeypatch.setattr(ds_mod.SensorDataset, "_build_from_parquet", fake_build)

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    _write_dummy_parquet(parquet_dir / "iobt_audio_polaris0150pm_rs0.parquet")
    cache_dir = tmp_path / "cache"

    ds_mod.SensorDataset(str(parquet_dir), CRLConfig(), cache_dir=cache_dir)
    ds_mod.SensorDataset(str(parquet_dir), CRLConfig(), cache_dir=cache_dir)
    assert len(built) == 2  # no caching — builds every time
