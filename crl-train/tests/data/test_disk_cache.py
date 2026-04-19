# tests/data/test_disk_cache.py
import hashlib
import time
from pathlib import Path
import numpy as np
import pytest

from crl_vehicle.data.dataset import SensorDataset, _compute_dir_hash
from crl_vehicle.config import CRLConfig


def test_cache_key_stable(tmp_path):
    """Same directory contents → same hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"y")
    h1 = _compute_dir_hash(tmp_path)
    h2 = _compute_dir_hash(tmp_path)
    assert h1 == h2


def test_cache_key_changes_on_new_file(tmp_path):
    """Adding a file → different hash."""
    (tmp_path / "a.parquet").write_bytes(b"x")
    h1 = _compute_dir_hash(tmp_path)
    (tmp_path / "b.parquet").write_bytes(b"y")
    h2 = _compute_dir_hash(tmp_path)
    assert h1 != h2


def test_cache_key_changes_on_mtime(tmp_path):
    """Touching a file → different hash."""
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    h1 = _compute_dir_hash(tmp_path)
    time.sleep(0.01)
    f.touch()
    h2 = _compute_dir_hash(tmp_path)
    assert h1 != h2


def _make_minimal_cache(tmp_path):
    """Build minimal _cache and _index structures matching SensorDataset internals."""
    audio_data = np.zeros((1, 16000), dtype=np.float32)
    seismic_data = np.zeros((1, 200), dtype=np.float32)
    cache = {
        "audio":   {("stem_a", None): {"data": audio_data,   "present": np.ones(16000, dtype=bool), "native_sr": 16000, "target_sr": 16000}},
        "seismic": {("stem_s", None): {"data": seismic_data, "present": np.ones(200,   dtype=bool), "native_sr": 200,   "target_sr": 200}},
    }
    index = [(("ds", "veh", "rs", None), 0, 0, 1, 0, 0)]
    groups = {("ds", "veh", "rs", None): {
        "audio_stem": "stem_a", "seismic_stem": "stem_s", "seg_key": None,
        "audio_nw": 1, "seismic_nw": 1, "vehicle_type": 0,
        "audio_seg_id": 0, "seismic_seg_id": 0,
    }}
    segment_id_map = {("audio", "stem_a", None): 0, ("seismic", "stem_s", None): 0}
    seg_counter = 1
    return cache, index, groups, segment_id_map, seg_counter


def test_save_cache_creates_files(tmp_path):
    from crl_vehicle.data.dataset import _save_cache
    cache, index, groups, sim, sc = _make_minimal_cache(tmp_path)
    cache_dir = tmp_path / "cache"
    _save_cache(cache_dir, "abc123", cache, index, groups, sim, sc)
    slot = cache_dir / "abc123"
    assert (slot / "audio.npy").exists()
    assert (slot / "seismic.npy").exists()
    assert (slot / "audio_meta.pkl").exists()
    assert (slot / "seismic_meta.pkl").exists()
    assert (slot / "index.pkl").exists()
    assert (slot / "groups.pkl").exists()
    assert (slot / "segment_id_map.pkl").exists()


def test_roundtrip_cache(tmp_path):
    from crl_vehicle.data.dataset import _save_cache, _load_cache
    cache, index, groups, sim, sc = _make_minimal_cache(tmp_path)
    cache_dir = tmp_path / "cache"
    _save_cache(cache_dir, "abc123", cache, index, groups, sim, sc)

    loaded_cache, loaded_index, loaded_groups, loaded_sim, loaded_sc = \
        _load_cache(cache_dir, "abc123")

    orig = cache["audio"][("stem_a", None)]["data"]
    loaded = loaded_cache["audio"][("stem_a", None)]["data"]
    np.testing.assert_array_equal(orig, loaded)

    assert loaded_index == index
    assert loaded_groups == groups
    assert loaded_sim == sim
    assert loaded_sc == sc


def test_load_cache_returns_none_for_missing_key(tmp_path):
    from crl_vehicle.data.dataset import _load_cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    result = _load_cache(cache_dir, "missing")
    assert result is None
