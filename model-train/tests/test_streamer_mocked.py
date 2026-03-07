import pytest
import torch
import config

from train import VehicleStreamer, extract_instance_from_table, assign_label


# ============================================================
# 1. Mock DB functions
# ============================================================


class FakeCursor:
    def __init__(self, tables):
        self.tables = tables
        self.last_query = None

    def execute(self, query):
        self.last_query = query

    def fetchall(self):
        # Return table names as rows
        return [(t,) for t in self.tables]


class FakeConnection:
    def __init__(self, tables):
        self.cursor_obj = FakeCursor(tables)

    def cursor(self):
        return self.cursor_obj


def fake_db_connect():
    # Fake tables across datasets
    tables = [
        "iobt_audio_polaris0150pm_rs1",
        "iobt_seismic_polaris0150pm_rs1",
        "focal_audio_walk_rs2",
        "focal_seismic_walk_rs2",
        "m3nvc_accel_cx30_rs3",
        "m3nvc_audio_cx30_rs3",
    ]
    conn = FakeConnection(tables)
    return conn, conn.cursor_obj


def fake_get_time_bounds(cursor, table_name, run_id=None):
    # Always return a fixed time range
    return (0.0, 30.0)  # 30 seconds available


def fake_fetch_sensor_batch(cursor, table_name, n_samples, start_time, run_id=None):
    # Return deterministic fake waveform data
    # n_samples is the number of samples requested
    return [(0.5,) for _ in range(n_samples)]


# ============================================================
# 2. Test the streamer with mocks
# ============================================================


def test_streamer_with_mocked_db(monkeypatch):
    # Patch DB functions
    monkeypatch.setattr("train.db_connect", fake_db_connect)
    monkeypatch.setattr("train.get_time_bounds", fake_get_time_bounds)
    monkeypatch.setattr("train.fetch_sensor_batch", fake_fetch_sensor_batch)

    # Force deterministic behavior
    monkeypatch.setattr(config, "SPLIT_TRAIN", 1.0)  # always train split
    monkeypatch.setattr(config, "SPLIT_VAL", 0.0)
    monkeypatch.setattr(config, "SPLIT_TEST", 0.0)

    # Use category mode for predictable labels
    monkeypatch.setattr(config, "TRAINING_MODE", "category")

    streamer = VehicleStreamer(split="train")
    iterator = iter(streamer)

    # Pull a few samples
    for _ in range(5):
        window, label = next(iterator)

        # -----------------------------
        # Validate window shape
        # -----------------------------
        assert isinstance(window, torch.Tensor)
        assert window.dim() == 2  # [C, T]
        assert (
            window.shape[1]
            == int(config.NATIVE_SR["iobt"]["audio"] * config.SAMPLE_SECONDS)
            or window.shape[1]
            == int(config.NATIVE_SR["focal"]["audio"] * config.SAMPLE_SECONDS)
            or window.shape[1]
            == int(config.NATIVE_SR["m3nvc"]["audio"] * config.SAMPLE_SECONDS)
        )

        # -----------------------------
        # Validate label correctness
        # -----------------------------
        assert isinstance(label, int)
        assert 0 <= label < config.NUM_CLASSES


# ============================================================
# 3. Test instance extraction on mocked tables
# ============================================================


@pytest.mark.parametrize(
    "table,expected",
    [
        ("iobt_audio_polaris0150pm_rs1", "polaris0150pm"),
        ("focal_seismic_walk_rs2", "walk"),
        ("m3nvc_accel_cx30_rs3", "cx30"),
    ],
)
def test_instance_extraction(table, expected):
    assert extract_instance_from_table(table) == expected


# ============================================================
# 4. Test label assignment on mocked instances
# ============================================================


def test_label_assignment_mocked(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "category")

    assert assign_label("iobt", "polaris0150pm") == 1
    assert assign_label("focal", "walk") == 0
    assert assign_label("m3nvc", "cx30") == 2
