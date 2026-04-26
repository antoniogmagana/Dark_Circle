"""Integration tests: SensorDataset with use_id_split=True."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset


# Smaller test window sizes to keep parquet fixtures tiny
TEST_AUDIO_W   = 16
TEST_SEISMIC_W = 4


@pytest.fixture(autouse=True)
def _patch_source_rates(monkeypatch):
    """Override _SOURCE_RATES (in both dataset.py and id_split.py) to match the
    tiny test window sizes. Without this, the loader would look up the real
    iobt source rate (16000 audio / 100 seismic) and try to reshape tiny test
    parquets at the wrong rate."""
    from crl_vehicle.data import dataset as ds_mod
    from crl_vehicle.data import id_split as id_mod
    fake_rates = {
        "iobt":  {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
        "focal": {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
        "m3nvc": {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
    }
    monkeypatch.setattr(ds_mod, "_SOURCE_RATES", fake_rates)
    monkeypatch.setattr(id_mod, "_SOURCE_RATES", fake_rates)


@pytest.fixture
def small_cfg():
    """CRLConfig with tiny window sizes so fixtures are fast."""
    cfg = CRLConfig()
    # Override modality_cfg to use small windows
    def patched(sensor):
        from crl_vehicle.config import ModalityConfig
        if sensor == "audio":
            return ModalityConfig(sample_rate=TEST_AUDIO_W, window_size=TEST_AUDIO_W, n_channels=1)
        return ModalityConfig(sample_rate=TEST_SEISMIC_W, window_size=TEST_SEISMIC_W, n_channels=1)
    cfg.modality_cfg = patched
    return cfg


def _write_simple_parquet(path, n_samples):
    df = pd.DataFrame({
        "amplitude": np.zeros(n_samples, dtype="float32"),
        "present":   np.ones(n_samples, dtype=bool),
    })
    df.to_parquet(path, index=False)


def _write_split_runs_parquet(path, scene_run_lengths):
    rows = []
    for s, r, n in scene_run_lengths:
        for _ in range(n):
            rows.append({"scene_id": s, "run_id": r,
                         "amplitude": 0.0, "present": True})
    df = pd.DataFrame(rows)
    df["amplitude"] = df["amplitude"].astype("float32")
    df["present"]   = df["present"].astype(bool)
    df["scene_id"]  = df["scene_id"].astype("int64")
    df["run_id"]    = df["run_id"].astype("int64")
    df.to_parquet(path, index=False)


@pytest.fixture
def id_root_split(tmp_path, monkeypatch):
    """Layout: data/{train,val,test}/iobt_{audio,seismic}_silverado_rs1.parquet
    with marker = "split". 10 windows in each sensor."""
    from crl_vehicle import config as cfg_mod
    monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "iobt",
        {"silverado": ["heavy", "pickup", "split"]})

    data_root = tmp_path / "data"
    for sub in ("train", "val", "test"):
        d = data_root / sub
        d.mkdir(parents=True)
    # Put the actual file in train/ (any subdir works under ID schema)
    _write_simple_parquet(data_root / "train" / "iobt_audio_silverado_rs1.parquet",
                          n_samples=TEST_AUDIO_W * 10)
    _write_simple_parquet(data_root / "train" / "iobt_seismic_silverado_rs1.parquet",
                          n_samples=TEST_SEISMIC_W * 10)
    return data_root


class TestIdSplitDatasetRoles:
    def test_train_role_is_empty_for_split_marker(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "train",   # ignored under id split
            config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="train",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        # "split" files contribute zero windows to train
        assert len(ds) == 0

    def test_val_role_gets_first_half(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "val",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        # 10 windows total, half = 5 → val gets indices 0..4
        assert len(ds) == 5
        # Verify w indices are 0..4
        windows = sorted({entry[1] for entry in ds._index})
        assert windows == [0, 1, 2, 3, 4]

    def test_test_role_gets_second_half(self, id_root_split, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_split / "test",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="test",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        assert len(ds) == 5
        windows = sorted({entry[1] for entry in ds._index})
        assert windows == [5, 6, 7, 8, 9]

    def test_val_and_test_indices_disjoint(self, id_root_split, tmp_path, small_cfg):
        val_ds = SensorDataset(
            parquet_dir=id_root_split / "val",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        test_ds = SensorDataset(
            parquet_dir=id_root_split / "test",
            config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="test",
            id_root=id_root_split,
            id_cache_dir=tmp_path / "id_cache",
        )
        v = {entry[1] for entry in val_ds._index}
        t = {entry[1] for entry in test_ds._index}
        assert v.isdisjoint(t)


class TestIdSplitDefaultOffUnchanged:
    def test_use_id_split_false_does_not_require_new_kwargs(
        self, id_root_split, tmp_path, small_cfg
    ):
        # With the flag off, behavior should match today exactly —
        # _index is built solely from parquet_dir, ignoring DATASET_VEHICLE_MAP marker.
        ds = SensorDataset(
            parquet_dir=id_root_split / "train",
            config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
        )
        # The "split" marker has no effect when use_id_split=False
        assert len(ds) == 10  # all 10 windows from the file


class TestIdSplitTrainMarker:
    def test_train_marker_routes_all_to_train(self, tmp_path, small_cfg, monkeypatch):
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "iobt",
            {"polaris": ["light", "polaris", "train"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        _write_simple_parquet(data_root / "train" / "iobt_audio_polaris_rs1.parquet",
                              n_samples=TEST_AUDIO_W * 6)
        _write_simple_parquet(data_root / "train" / "iobt_seismic_polaris_rs1.parquet",
                              n_samples=TEST_SEISMIC_W * 6)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        train_ds = SensorDataset(
            parquet_dir=data_root / "train", config=small_cfg, is_train=True,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="train",
            id_root=data_root, id_cache_dir=tmp_path / "id_cache",
        )
        assert len(train_ds) == 6

        val_ds = SensorDataset(
            parquet_dir=data_root / "val", config=small_cfg, is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True, role="val",
            id_root=data_root, id_cache_dir=tmp_path / "id_cache",
        )
        assert len(val_ds) == 0


class TestIdSplitRunsMarker:
    def test_split_runs_partitions_across_three_roles(self, tmp_path, small_cfg, monkeypatch):
        """End-to-end split_runs: 3 paired runs → train/val/test all non-empty."""
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "m3nvc",
            {"cx30": ["medium", "cx30", "split_runs"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        # Three runs, each large enough to yield a few windows after ceil/floor.
        # Audio window=16, seismic window=4. Each run = 64 audio samples + 16 seismic = 4 windows.
        audio_runs   = [(1, 6, TEST_AUDIO_W * 4), (1, 7, TEST_AUDIO_W * 4),
                        (2, 6, TEST_AUDIO_W * 4)]
        seismic_runs = [(1, 6, TEST_SEISMIC_W * 4), (1, 7, TEST_SEISMIC_W * 4),
                        (2, 6, TEST_SEISMIC_W * 4)]
        _write_split_runs_parquet(data_root / "train" / "m3nvc_audio_cx30_rs1.parquet",
                                  audio_runs)
        _write_split_runs_parquet(data_root / "train" / "m3nvc_seismic_cx30_rs1.parquet",
                                  seismic_runs)

        lengths = {}
        for role in ("train", "val", "test"):
            ds = SensorDataset(
                parquet_dir=data_root / role, config=small_cfg, is_train=(role == "train"),
                cache_dir=tmp_path / f"raw_cache_{role}",
                use_id_split=True, role=role,
                id_root=data_root, id_cache_dir=tmp_path / "id_cache",
            )
            lengths[role] = len(ds)
        # All three roles must have ≥1 window (floor)
        assert lengths["train"] >= 1
        assert lengths["val"]   >= 1
        assert lengths["test"]  >= 1
        # Total windows = 12 (3 runs × 4 windows each, with whole-run boundaries)
        # But ceil/floor may shave a few edges — accept 9..12.
        assert 9 <= sum(lengths.values()) <= 12

    def test_split_runs_dropped_run_warning(self, tmp_path, small_cfg, monkeypatch, caplog):
        """A (scene, run) present only in one sensor is dropped with WARN."""
        import logging
        from crl_vehicle import config as cfg_mod
        monkeypatch.setitem(cfg_mod.DATASET_VEHICLE_MAP, "m3nvc",
            {"cx30": ["medium", "cx30", "split_runs"]})
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        for sub in ("val", "test"):
            (data_root / sub).mkdir()

        # Audio has runs (1,6), (1,7), (2,6); seismic has (1,6), (1,7) only.
        # (2,6) should be dropped with reason="single_sensor".
        _write_split_runs_parquet(
            data_root / "train" / "m3nvc_audio_cx30_rs1.parquet",
            [(1, 6, TEST_AUDIO_W * 4), (1, 7, TEST_AUDIO_W * 4),
             (2, 6, TEST_AUDIO_W * 4)],
        )
        _write_split_runs_parquet(
            data_root / "train" / "m3nvc_seismic_cx30_rs1.parquet",
            [(1, 6, TEST_SEISMIC_W * 4), (1, 7, TEST_SEISMIC_W * 4)],
        )

        with caplog.at_level(logging.WARNING):
            SensorDataset(
                parquet_dir=data_root / "train", config=small_cfg, is_train=True,
                cache_dir=tmp_path / "raw_cache",
                use_id_split=True, role="train",
                id_root=data_root, id_cache_dir=tmp_path / "id_cache",
            )
        # WARN must mention the dropped (scene, run) and the reason
        warn_text = " ".join(r.message for r in caplog.records)
        assert "(2, 6)" in warn_text
        assert "single_sensor" in warn_text


import sys


class TestIdSplitCli:
    def test_use_id_split_flag_parsed(self, monkeypatch):
        # Import train.py's parser directly
        sys.path.insert(0, str(Path(__file__).parents[2]))  # crl-train/
        import train
        monkeypatch.setattr("sys.argv", ["train.py", "--use-id-split", "--id-root", "/tmp/x"])
        args = train.parse_args()
        assert args.use_id_split is True
        assert args.id_root == "/tmp/x"

    def test_use_id_split_default_false(self, monkeypatch):
        sys.path.insert(0, str(Path(__file__).parents[2]))
        import train
        monkeypatch.setattr("sys.argv", ["train.py"])
        args = train.parse_args()
        assert args.use_id_split is False
