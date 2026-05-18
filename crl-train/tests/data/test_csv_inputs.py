"""End-to-end tests: SensorDataset and id_split accept .csv inputs.

These tests verify the dispatcher is threaded into the production code paths:
- non-id-split mode loads .csv from parquet_dir
- id-split mode loads .csv from id_root subdirs
- mixed .parquet + .csv directory works (parquet wins on stem conflict)
- split_runs marker works for CSV (scene_id/run_id columns)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import SensorDataset

# Mirror tiny window sizes from test_id_split_dataset.py
TEST_AUDIO_W = 16
TEST_SEISMIC_W = 4


@pytest.fixture(autouse=True)
def _patch_source_rates(monkeypatch):
    from crl_vehicle.data import dataset as ds_mod
    from crl_vehicle.data import id_split as id_mod

    fake_rates = {
        "iobt": {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
        "focal": {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
        "m3nvc": {"audio": TEST_AUDIO_W, "seismic": TEST_SEISMIC_W},
    }
    monkeypatch.setattr(ds_mod, "_SOURCE_RATES", fake_rates)
    monkeypatch.setattr(id_mod, "_SOURCE_RATES", fake_rates)


@pytest.fixture
def small_cfg():
    cfg = CRLConfig()

    def patched(sensor):
        from crl_vehicle.config import ModalityConfig

        if sensor == "audio":
            return ModalityConfig(sample_rate=TEST_AUDIO_W, window_size=TEST_AUDIO_W, n_channels=1)
        return ModalityConfig(sample_rate=TEST_SEISMIC_W, window_size=TEST_SEISMIC_W, n_channels=1)

    cfg.modality_cfg = patched
    return cfg


def _write_csv(path: Path, n_samples: int) -> None:
    df = pd.DataFrame(
        {
            "amplitude": np.zeros(n_samples, dtype="float32"),
            "present": np.ones(n_samples, dtype=bool),
        }
    )
    df.to_csv(path, index=False)


def _write_parquet(path: Path, n_samples: int) -> None:
    df = pd.DataFrame(
        {
            "amplitude": np.zeros(n_samples, dtype="float32"),
            "present": np.ones(n_samples, dtype=bool),
        }
    )
    df.to_parquet(path, index=False)


def _write_split_runs_csv(path: Path, scene_run_lengths: list[tuple[int, int, int]]) -> None:
    rows = []
    for s, r, n in scene_run_lengths:
        rows.extend([(s, r, 0.0, True)] * n)
    df = pd.DataFrame(rows, columns=["scene_id", "run_id", "amplitude", "present"])
    df["amplitude"] = df["amplitude"].astype("float32")
    df["present"] = df["present"].astype(bool)
    df["scene_id"] = df["scene_id"].astype("int64")
    df["run_id"] = df["run_id"].astype("int64")
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Non-id-split mode: CSV-only directory
# ---------------------------------------------------------------------------


class TestCsvOnlyDirectory:
    @pytest.fixture
    def csv_root(self, tmp_path, monkeypatch):
        from crl_vehicle import config as cfg_mod

        monkeypatch.setitem(
            cfg_mod.DATASET_VEHICLE_MAP, "iobt", {"silverado": ["heavy", "pickup", "train"]}
        )
        d = tmp_path / "data"
        d.mkdir()
        _write_csv(d / "iobt_audio_silverado_rs1.csv", n_samples=TEST_AUDIO_W * 5)
        _write_csv(d / "iobt_seismic_silverado_rs1.csv", n_samples=TEST_SEISMIC_W * 5)
        return d

    def test_csv_only_directory_loads(self, csv_root, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=csv_root,
            config=small_cfg,
            is_train=True,
            cache_dir=tmp_path / "raw_cache",
        )
        assert len(ds) == 5

    def test_csv_only_yields_proper_windows(self, csv_root, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=csv_root,
            config=small_cfg,
            is_train=True,
            cache_dir=tmp_path / "raw_cache",
        )
        item = ds[0]
        assert item["x_audio"].shape == (1, TEST_AUDIO_W)
        assert item["x_seismic"].shape == (1, TEST_SEISMIC_W)


# ---------------------------------------------------------------------------
# Non-id-split mode: mixed parquet + CSV directory
# ---------------------------------------------------------------------------


class TestMixedDirectoryParquetWins:
    @pytest.fixture
    def mixed_root(self, tmp_path, monkeypatch):
        from crl_vehicle import config as cfg_mod

        monkeypatch.setitem(
            cfg_mod.DATASET_VEHICLE_MAP,
            "iobt",
            {
                "silverado": ["heavy", "pickup", "train"],
                "polaris": ["medium", "atv", "train"],
            },
        )
        d = tmp_path / "data"
        d.mkdir()
        # silverado: parquet only
        _write_parquet(d / "iobt_audio_silverado_rs1.parquet", n_samples=TEST_AUDIO_W * 3)
        _write_parquet(d / "iobt_seismic_silverado_rs1.parquet", n_samples=TEST_SEISMIC_W * 3)
        # polaris: csv only
        _write_csv(d / "iobt_audio_polaris_rs1.csv", n_samples=TEST_AUDIO_W * 4)
        _write_csv(d / "iobt_seismic_polaris_rs1.csv", n_samples=TEST_SEISMIC_W * 4)
        return d

    def test_mixed_loads_both_groups(self, mixed_root, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=mixed_root,
            config=small_cfg,
            is_train=True,
            cache_dir=tmp_path / "raw_cache",
        )
        # 3 silverado windows + 4 polaris windows = 7
        assert len(ds) == 7

    def test_parquet_wins_on_stem_conflict(
        self, mixed_root, tmp_path, small_cfg, caplog
    ):
        # Add a CSV alongside an existing parquet — parquet must win and
        # a warning must be emitted.
        import logging

        _write_csv(
            mixed_root / "iobt_audio_silverado_rs1.csv", n_samples=TEST_AUDIO_W * 99
        )

        with caplog.at_level(logging.WARNING):
            ds = SensorDataset(
                parquet_dir=mixed_root,
                config=small_cfg,
                is_train=True,
                cache_dir=tmp_path / "raw_cache",
            )

        # If the CSV had won, silverado would contribute 99 windows. The
        # parquet has 3. Total: 3 silverado + 4 polaris = 7.
        assert len(ds) == 7
        assert any(
            "iobt_audio_silverado_rs1" in rec.message and "ignoring CSV" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# id-split mode: CSV under id_root subdirs
# ---------------------------------------------------------------------------


class TestIdSplitCsv:
    @pytest.fixture
    def id_root_csv(self, tmp_path, monkeypatch):
        from crl_vehicle import config as cfg_mod

        monkeypatch.setitem(
            cfg_mod.DATASET_VEHICLE_MAP, "iobt", {"silverado": ["heavy", "pickup", "split"]}
        )
        data_root = tmp_path / "data"
        for sub in ("train", "val", "test"):
            (data_root / sub).mkdir(parents=True)
        # split marker — files in train/, manifest splits into val/test halves.
        _write_csv(
            data_root / "train" / "iobt_audio_silverado_rs1.csv",
            n_samples=TEST_AUDIO_W * 10,
        )
        _write_csv(
            data_root / "train" / "iobt_seismic_silverado_rs1.csv",
            n_samples=TEST_SEISMIC_W * 10,
        )
        return data_root

    def test_id_split_loads_csv_val_role(self, id_root_csv, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_csv / "train",  # ignored under id split
            config=small_cfg,
            is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True,
            role="val",
            id_root=id_root_csv,
            id_cache_dir=tmp_path / "id_cache",
        )
        # 10 windows split half/half → 5 in val
        assert len(ds) == 5

    def test_id_split_csv_test_role(self, id_root_csv, tmp_path, small_cfg):
        ds = SensorDataset(
            parquet_dir=id_root_csv / "train",
            config=small_cfg,
            is_train=False,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True,
            role="test",
            id_root=id_root_csv,
            id_cache_dir=tmp_path / "id_cache",
        )
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# id-split mode: split_runs marker on CSV (scene_id/run_id columns)
# ---------------------------------------------------------------------------


class TestIdSplitRunsCsv:
    @pytest.fixture
    def id_root_split_runs_csv(self, tmp_path, monkeypatch):
        from crl_vehicle import config as cfg_mod

        monkeypatch.setitem(
            cfg_mod.DATASET_VEHICLE_MAP, "m3nvc", {"cx30": ["medium", "cx30", "split_runs"]}
        )
        data_root = tmp_path / "data"
        (data_root / "train").mkdir(parents=True)
        # 4 runs across 2 scenes — large enough for a 50/25/25 split.
        audio_runs = [(0, 0, TEST_AUDIO_W * 3), (0, 1, TEST_AUDIO_W * 3),
                      (1, 0, TEST_AUDIO_W * 3), (1, 1, TEST_AUDIO_W * 3)]
        seismic_runs = [(0, 0, TEST_SEISMIC_W * 3), (0, 1, TEST_SEISMIC_W * 3),
                        (1, 0, TEST_SEISMIC_W * 3), (1, 1, TEST_SEISMIC_W * 3)]
        _write_split_runs_csv(
            data_root / "train" / "m3nvc_audio_cx30_rs1.csv", audio_runs
        )
        _write_split_runs_csv(
            data_root / "train" / "m3nvc_seismic_cx30_rs1.csv", seismic_runs
        )
        return data_root

    def test_split_runs_csv_loads(self, id_root_split_runs_csv, tmp_path, small_cfg):
        # Building the manifest requires reading scene_id/run_id from CSV.
        # This is the test that proves id_split's extract_runs handles CSV.
        ds = SensorDataset(
            parquet_dir=id_root_split_runs_csv / "train",
            config=small_cfg,
            is_train=True,
            cache_dir=tmp_path / "raw_cache",
            use_id_split=True,
            role="train",
            id_root=id_root_split_runs_csv,
            id_cache_dir=tmp_path / "id_cache",
        )
        # 12 windows total (4 runs × 3 windows); 50% train → roughly 6.
        # Exact count depends on partition; assert it's non-empty.
        assert len(ds) > 0
