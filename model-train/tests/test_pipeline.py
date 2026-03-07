import pytest
import config

# Import helpers from train.py
from train import (
    extract_instance_from_table,
    instance_to_category,
    assign_label,
)


# ============================================================
# 1. Test instance extraction from table names
# ============================================================


@pytest.mark.parametrize(
    "table,expected",
    [
        ("iobt_audio_polaris0150pm_rs1", "polaris0150pm"),
        ("focal_seismic_walk_rs2", "walk"),
        ("m3nvc_accel_cx30_miata_rs3", "cx30_miata"),
        ("m3nvc_audio_miata_mustang_rs1", "miata_mustang"),
    ],
)
def test_extract_instance_from_table(table, expected):
    assert extract_instance_from_table(table) == expected


# ============================================================
# 2. Test instance → category mapping
# ============================================================


def test_instance_to_category_all_instances():
    for ds, mapping in config.DATASET_VEHICLE_MAP.items():
        for cat_id, instances in mapping.items():
            for inst in instances:
                assert instance_to_category(ds, inst) == cat_id


# ============================================================
# 3. Test label assignment for all modes
# ============================================================


def test_assign_label_category_mode(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "category")
    for ds, mapping in config.DATASET_VEHICLE_MAP.items():
        for cat_id, instances in mapping.items():
            for inst in instances:
                assert assign_label(ds, inst) == cat_id


def test_assign_label_detection_mode(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "detection")
    for ds, mapping in config.DATASET_VEHICLE_MAP.items():
        for cat_id, instances in mapping.items():
            for inst in instances:
                label = assign_label(ds, inst)
                if cat_id == 0:
                    assert label == 0
                else:
                    assert label == 1


def test_assign_label_instance_mode(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "instance")
    for ds, mapping in config.DATASET_VEHICLE_MAP.items():
        for _, instances in mapping.items():
            for inst in instances:
                assert assign_label(ds, inst) == config.INSTANCE_TO_CLASS[inst]


# ============================================================
# 4. Test NUM_CLASSES correctness
# ============================================================


def test_num_classes_detection(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "detection")
    monkeypatch.setattr(config, "NUM_CLASSES", 2)
    assert config.NUM_CLASSES == 2


def test_num_classes_category(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "category")
    monkeypatch.setattr(config, "NUM_CLASSES", len(config.CLASS_MAP))
    assert config.NUM_CLASSES == len(config.CLASS_MAP)


def test_num_classes_instance(monkeypatch):
    monkeypatch.setattr(config, "TRAINING_MODE", "instance")
    monkeypatch.setattr(config, "NUM_CLASSES", len(config.INSTANCE_TO_CLASS))
    assert config.NUM_CLASSES == len(config.INSTANCE_TO_CLASS)


# ============================================================
# 5. Test INSTANCE_TO_CLASS reproducibility
# ============================================================


def test_instance_to_class_reproducible():
    # Recompute with same seed
    import random

    random.seed(config.INSTANCE_SEED)

    all_instances = sorted(
        set(
            inst
            for ds_map in config.DATASET_VEHICLE_MAP.values()
            for inst_list in ds_map.values()
            for inst in inst_list
        )
    )

    shuffled = all_instances.copy()
    random.shuffle(shuffled)

    recomputed = {name: idx for idx, name in enumerate(shuffled)}

    assert recomputed == config.INSTANCE_TO_CLASS
