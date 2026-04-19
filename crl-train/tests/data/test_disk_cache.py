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
