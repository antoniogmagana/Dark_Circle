"""Unit tests for ID split manifest builder."""
from crl_vehicle.config import CRLConfig


def test_use_id_split_field_defaults_false():
    cfg = CRLConfig()
    assert cfg.use_id_split is False


def test_use_id_split_field_settable():
    cfg = CRLConfig(use_id_split=True)
    assert cfg.use_id_split is True
