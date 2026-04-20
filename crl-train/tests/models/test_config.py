import pytest
from crl_vehicle.config import CRLConfig, ModalityConfig


def test_modality_config_fields():
    mc = ModalityConfig(sample_rate=16000, window_size=16000, n_channels=1)
    assert mc.sample_rate == 16000
    assert mc.window_size == 16000
    assert mc.n_channels == 1


def test_crlconfig_defaults():
    cfg = CRLConfig()
    assert cfg.d_z == 24
    assert cfg.frontend_type == "multiscale"
    assert cfg.fused_seq_len == 32
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.n_layers == 2


def test_modality_cfg_audio():
    cfg = CRLConfig()
    mc = cfg.modality_cfg("audio")
    assert mc.sample_rate == 16000
    assert mc.window_size == 16000
    assert mc.n_channels == 1


def test_modality_cfg_seismic():
    cfg = CRLConfig()
    mc = cfg.modality_cfg("seismic")
    assert mc.sample_rate == 200
    assert mc.window_size == 200
    assert mc.n_channels == 1


def test_modality_cfg_unknown_raises():
    cfg = CRLConfig()
    with pytest.raises(ValueError, match="Unknown modality"):
        cfg.modality_cfg("lidar")


def test_loss_weights():
    cfg = CRLConfig()
    assert cfg.lambda_interv == 1.0
    assert cfg.lambda_aux_pres == 1.0
    assert cfg.lambda_aux_type == 1.0
    assert cfg.lambda_aux_prox == 0.1


def test_kl_schedule_params():
    cfg = CRLConfig()
    assert cfg.kl_floor == 0.01
    assert cfg.kl_target == 0.5
    assert cfg.beta_step == 0.02


def test_training_params():
    cfg = CRLConfig()
    assert cfg.batch_size == 128
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.wd == pytest.approx(1e-4)
    assert cfg.n_epochs == 100
    assert cfg.early_stop_patience == 25


def test_partner_counts():
    cfg = CRLConfig()
    assert cfg.n_partners_same_type == 1
    assert cfg.n_partners_diff_type == 1
    assert cfg.n_partners_cross_ds == 1


def test_fused_seq_len_configurable():
    cfg = CRLConfig(fused_seq_len=64)
    assert cfg.fused_seq_len == 64
