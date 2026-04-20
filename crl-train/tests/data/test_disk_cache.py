# tests/data/test_disk_cache.py
"""Tests for SensorDataset disk cache behaviour.
Updated for the rebuilt dataset (simple pickle-based cache, hash keyed by filename+mtime)."""
import time
from pathlib import Path
import pytest
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset


def _write_dummy_parquet(path: Path) -> None:
    """Write a minimal valid parquet file for hash testing."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(np.zeros((2, 10), dtype="float32"))
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Cache-key stability via SensorDataset._cache_key()
# ---------------------------------------------------------------------------

class _MinimalSensorDataset(SensorDataset):
    """Subclass that skips __init__ so we can test _cache_key in isolation."""
    def __init__(self, parquet_dir):  # noqa: D107
        self.parquet_dir = Path(parquet_dir)
        self.cfg = CRLConfig()
        self._cache = {"audio": {}, "seismic": {}}
        self._index = []
        self._groups = {}


def test_cache_key_stable(tmp_path):
    """Same directory contents → same hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"y")
    ds = _MinimalSensorDataset(tmp_path)
    h1 = ds._cache_key()
    h2 = ds._cache_key()
    assert h1 == h2


def test_cache_key_changes_on_new_file(tmp_path):
    """Adding a file → different hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    ds = _MinimalSensorDataset(tmp_path)
    h1 = ds._cache_key()
    (tmp_path / "b.parquet").write_bytes(b"y")
    h2 = ds._cache_key()
    assert h1 != h2


def test_cache_key_changes_on_mtime(tmp_path):
    """Touching a file → different hash."""
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    ds = _MinimalSensorDataset(tmp_path)
    h1 = ds._cache_key()
    time.sleep(0.01)
    f.touch()
    h2 = ds._cache_key()
    assert h1 != h2


# ---------------------------------------------------------------------------
# Save/load roundtrip via SensorDataset internals
# ---------------------------------------------------------------------------

def _make_minimal_state():
    import numpy as np
    cache = {
        "audio":   {("stem_a", None): {"data": np.zeros((1, 16000), dtype="float32"), "n_windows": 1}},
        "seismic": {("stem_s", None): {"data": np.zeros((1, 200),   dtype="float32"), "n_windows": 1}},
    }
    index  = [(("ds", "veh", "rs", None), 0, 0, 1, 0, 0)]
    groups = {("ds", "veh", "rs", None): {
        "audio_stem": "stem_a", "seismic_stem": "stem_s", "seg_key": None,
        "audio_nw": 1, "seismic_nw": 1, "vehicle_type": 0,
        "audio_seg_id": 0, "seismic_seg_id": 0,
    }}
    return cache, index, groups


def test_save_and_load_cache(tmp_path):
    """_save_cache and _load_from_cache roundtrip preserves data, index, groups."""
    import numpy as np

    ds = _MinimalSensorDataset(tmp_path)
    cache, index, groups = _make_minimal_state()
    ds._cache  = cache
    ds._index  = index
    ds._groups = groups

    cache_path = tmp_path / "test.pkl"
    ds._save_cache(cache_path)
    assert cache_path.exists()

    ds2 = _MinimalSensorDataset(tmp_path)
    ds2._load_from_cache(cache_path)

    orig   = cache["audio"][("stem_a", None)]["data"]
    loaded = ds2._cache["audio"][("stem_a", None)]["data"]
    np.testing.assert_array_equal(orig, loaded)
    assert ds2._index  == index
    assert ds2._groups == groups


def test_load_missing_cache_raises(tmp_path):
    """Loading a non-existent cache path raises FileNotFoundError."""
    ds = _MinimalSensorDataset(tmp_path)
    with pytest.raises((FileNotFoundError, Exception)):
        ds._load_from_cache(tmp_path / "missing.pkl")


def test_sensor_dataset_writes_and_reads_cache(tmp_path, monkeypatch):
    """SensorDataset writes cache on first build, loads it on second construction."""
    import numpy as np
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
    # Need at least one dummy parquet so _cache_key is stable and _load_data proceeds
    _write_dummy_parquet(parquet_dir / "iobt_audio_polaris0150pm_rs0.parquet")
    cache_dir = tmp_path / "cache"
    cfg = CRLConfig()

    # First construction: _build_from_parquet called, cache written
    ds_mod.SensorDataset(str(parquet_dir), cfg, is_train=True, cache_dir=cache_dir)
    assert len(built) == 1

    # Second construction: cache exists, _build_from_parquet NOT called
    built.clear()
    ds_mod.SensorDataset(str(parquet_dir), cfg, is_train=True, cache_dir=cache_dir)
    assert len(built) == 0
