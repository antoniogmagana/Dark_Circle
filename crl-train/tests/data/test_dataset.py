"""Tests using MagicMock — no parquet files required."""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    StratifiedPairDataset, collate_pairs, collate_single,
    STRATUM_CONSEC, STRATUM_SAME_TYPE, STRATUM_DIFF_TYPE, STRATUM_CROSS_DS,
)


def _make_mock_sensor_ds(cfg=None):
    cfg = cfg or CRLConfig()
    ds = MagicMock()
    ds.cfg = cfg

    # 3 groups: iobt/polaris(light=1), iobt/silverado(utility=3), focal/motor(light=1)
    gkeys = [
        ("iobt",  "polaris0150pm",  "rs0", None),
        ("iobt",  "silverado0255pm", "rs0", None),
        ("focal", "motor",           "rs0", None),
    ]
    vtypes = [1, 3, 1]

    index = []
    groups = {}
    for seg_id, (gkey, vtype) in enumerate(zip(gkeys, vtypes)):
        ds_name, vehicle, rs, seg_key = gkey
        for w in range(3):
            index.append((gkey, w, vtype, 1, seg_id, seg_id))
        groups[gkey] = {
            "audio_stem":   f"{ds_name}_audio_{vehicle}_{rs}",
            "seismic_stem": f"{ds_name}_seismic_{vehicle}_{rs}",
            "seg_key": seg_key,
            "audio_nw": 3,
            "seismic_nw": 3,
            "vehicle_type": vtype,
            "audio_seg_id": seg_id,
            "seismic_seg_id": seg_id,
        }

    ds._index = index
    ds._groups = groups
    ds.is_train = True

    def _get_window(sensor, stem, seg_key, w, interv_idx):
        size = 16000 if sensor == "audio" else 200
        return torch.zeros(1, size)

    ds._get_window = _get_window
    return ds


class TestStratifiedPairDataset:

    @pytest.fixture
    def pair_ds(self):
        return StratifiedPairDataset(_make_mock_sensor_ds())

    def test_len_equals_consecutive_pairs(self, pair_ds):
        # 3 groups × 2 consecutive pairs each = 6
        assert len(pair_ds) == 6

    def test_anchor_keys_present(self, pair_ds):
        item = pair_ds[0]
        for key in ["x_audio_t", "x_seismic_t", "detection_label_t",
                    "vehicle_type_t", "audio_avail", "seismic_avail"]:
            assert key in item, f"Missing key: {key}"

    def test_partner_keys_present(self, pair_ds):
        item = pair_ds[0]
        assert "x_audio_p0" in item
        assert "x_seismic_p0" in item
        assert "partner_stratum_p0" in item

    def test_consecutive_partner_stratum(self, pair_ds):
        assert pair_ds[0]["partner_stratum_p0"] == STRATUM_CONSEC

    def test_all_strata_valid(self, pair_ds):
        valid = {STRATUM_CONSEC, STRATUM_SAME_TYPE, STRATUM_DIFF_TYPE, STRATUM_CROSS_DS}
        item = pair_ds[0]
        p = 0
        while f"partner_stratum_p{p}" in item:
            assert item[f"partner_stratum_p{p}"] in valid
            p += 1

    def test_audio_shape(self, pair_ds):
        assert pair_ds[0]["x_audio_t"].shape == (1, 16000)

    def test_seismic_shape(self, pair_ds):
        assert pair_ds[0]["x_seismic_t"].shape == (1, 200)

    def test_n_partners_matches_config(self, pair_ds):
        cfg = pair_ds.ds.cfg
        expected = 1 + cfg.n_partners_same_type + cfg.n_partners_diff_type + cfg.n_partners_cross_ds
        n = sum(1 for k in pair_ds[0] if k.startswith("x_audio_p"))
        assert n == expected

    def test_avail_flags_bool(self, pair_ds):
        item = pair_ds[0]
        assert isinstance(item["audio_avail"], bool)
        assert isinstance(item["seismic_avail"], bool)


class TestCollatePairs:

    def test_stacks_batch_dim(self):
        pair_ds = StratifiedPairDataset(_make_mock_sensor_ds())
        batch = [pair_ds[i] for i in range(2)]
        out = collate_pairs(batch)
        assert out["x_audio_t"].shape[0] == 2
        assert "n_partners" in out

    def test_n_partners_count(self):
        cfg = CRLConfig()
        pair_ds = StratifiedPairDataset(_make_mock_sensor_ds(cfg))
        out = collate_pairs([pair_ds[0]])
        expected = 1 + cfg.n_partners_same_type + cfg.n_partners_diff_type + cfg.n_partners_cross_ds
        assert out["n_partners"] == expected
