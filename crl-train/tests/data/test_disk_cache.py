# tests/data/test_disk_cache.py
"""Tests for SensorDataset mmap disk cache behaviour."""
import time
from pathlib import Path
import numpy as np
import pytest
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset


def _write_dummy_parquet(path: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(np.zeros((2, 10), dtype="float32"))
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Cache-key stability
# ---------------------------------------------------------------------------

class _MinimalSensorDataset(SensorDataset):
    """Subclass that skips __init__ so we can test internals in isolation."""
    def __init__(self, parquet_dir):
        self.parquet_dir = Path(parquet_dir)
        self.cfg = CRLConfig()
        self._cache = {"audio": {}, "seismic": {}}
        self._index = []
        self._groups = {}


def test_cache_key_stable(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"y")
    ds = _MinimalSensorDataset(tmp_path)
    assert ds._cache_key() == ds._cache_key()


def test_cache_key_changes_on_new_file(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x")
    ds = _MinimalSensorDataset(tmp_path)
    h1 = ds._cache_key()
    (tmp_path / "b.parquet").write_bytes(b"y")
    assert h1 != ds._cache_key()


def test_cache_key_changes_on_mtime(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    ds = _MinimalSensorDataset(tmp_path)
    h1 = ds._cache_key()
    time.sleep(0.01)
    f.touch()
    assert h1 != ds._cache_key()


# ---------------------------------------------------------------------------
# mmap save/load roundtrip
# ---------------------------------------------------------------------------

def _make_minimal_state():
    cache = {
        "audio":   {("stem_a", None): {"data": np.zeros((3, 16000), dtype="float32"), "n_windows": 3}},
        "seismic": {("stem_s", None): {"data": np.ones((2, 200),    dtype="float32"), "n_windows": 2}},
    }
    index  = [(("ds", "veh", "rs", None), 0, 0, 1, 0, 0)]
    groups = {("ds", "veh", "rs", None): {
        "audio_stem": "stem_a", "seismic_stem": "stem_s", "seg_key": None,
        "audio_nw": 3, "seismic_nw": 2, "vehicle_type": 0,
        "audio_seg_id": 0, "seismic_seg_id": 0,
    }}
    return cache, index, groups


def test_save_mmap_creates_files(tmp_path):
    ds = _MinimalSensorDataset(tmp_path)
    cache, index, groups = _make_minimal_state()
    ds._cache = cache
    ds._index = index
    ds._groups = groups

    cache_dir = tmp_path / "cache"
    ds._save_mmap_cache(cache_dir, "testhash")

    slot = cache_dir / "testhash"
    for name in ("audio.npy", "seismic.npy", "audio_meta.pkl", "seismic_meta.pkl",
                 "index.pkl", "groups.pkl"):
        assert (slot / name).exists(), f"missing {name}"


def test_mmap_roundtrip_data(tmp_path):
    ds = _MinimalSensorDataset(tmp_path)
    cache, index, groups = _make_minimal_state()
    ds._cache = cache
    ds._index = index
    ds._groups = groups

    cache_dir = tmp_path / "cache"
    ds._save_mmap_cache(cache_dir, "k1")

    ds2 = _MinimalSensorDataset(tmp_path)
    assert ds2._load_from_mmap_cache(cache_dir, "k1") is True

    orig   = cache["audio"][("stem_a", None)]["data"]
    loaded = ds2._cache["audio"][("stem_a", None)]["data"]
    np.testing.assert_array_equal(orig, loaded)
    assert loaded.shape == (3, 16000)

    orig_s   = cache["seismic"][("stem_s", None)]["data"]
    loaded_s = ds2._cache["seismic"][("stem_s", None)]["data"]
    np.testing.assert_array_equal(orig_s, loaded_s)
    assert loaded_s.shape == (2, 200)

    assert ds2._index  == index
    assert ds2._groups == groups


def test_mmap_load_returns_false_for_missing(tmp_path):
    ds = _MinimalSensorDataset(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    assert ds._load_from_mmap_cache(cache_dir, "nonexistent") is False


def test_mmap_views_are_writable_after_copy(tmp_path):
    """Windows extracted from mmap views must be writable (needed by remove_dc)."""
    ds = _MinimalSensorDataset(tmp_path)
    cache, index, groups = _make_minimal_state()
    ds._cache = cache
    ds._index = index
    ds._groups = groups

    cache_dir = tmp_path / "cache"
    ds._save_mmap_cache(cache_dir, "k2")

    ds2 = _MinimalSensorDataset(tmp_path)
    ds2._load_from_mmap_cache(cache_dir, "k2")

    row = ds2._cache["audio"][("stem_a", None)]["data"][0].copy()
    row[0] = 999.0  # must not raise


# ---------------------------------------------------------------------------
# Integration: SensorDataset writes on first build, loads on second
# ---------------------------------------------------------------------------

def test_sensor_dataset_writes_and_reads_cache(tmp_path, monkeypatch):
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

    ds_mod.SensorDataset(str(parquet_dir), CRLConfig(), is_train=True, cache_dir=cache_dir)
    assert len(built) == 1

    built.clear()
    ds_mod.SensorDataset(str(parquet_dir), CRLConfig(), is_train=True, cache_dir=cache_dir)
    assert len(built) == 0
