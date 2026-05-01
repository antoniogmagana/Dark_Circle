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
    assert mc.sample_rate == 100
    assert mc.window_size == 100
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


# ---------------------------------------------------------------------------
# New unified frontend schema (frontend_bank / frontend_fusion / params)
# ---------------------------------------------------------------------------


def test_frontend_schema_defaults():
    cfg = CRLConfig()
    assert cfg.frontend_bank == "multiscale"
    assert cfg.frontend_fusion == "early"
    assert cfg.audio_target_rate == 16000
    assert "audio" in cfg.frontend_per_sensor_params
    assert "seismic" in cfg.frontend_per_sensor_params
    assert cfg.frontend_per_sensor_params["audio"]["target_tokens"] == 32
    assert cfg.frontend_per_sensor_params["audio"]["kernel_sizes"] == [9, 19, 39, 159]


def test_audio_target_rate_propagates():
    cfg = CRLConfig(audio_target_rate=4000)
    mc = cfg.modality_cfg("audio")
    assert mc.sample_rate == 4000
    assert mc.window_size == 4000


def test_audio_target_rate_invalid_raises():
    with pytest.raises(ValueError, match="audio_target_rate"):
        CRLConfig(audio_target_rate=0)
    with pytest.raises(ValueError, match="audio_target_rate"):
        CRLConfig(audio_target_rate=-1)


@pytest.mark.parametrize(
    "legacy_type,expected_bank,expected_fusion",
    [
        ("morlet", "morlet", "late"),
        ("morlet_per_sensor", "morlet", "late"),
        ("morlet_fused", "morlet", "early"),
        ("morlet_learnable", "morlet_learnable", "late"),
        ("morlet_learnable_fused", "morlet_learnable", "early"),
    ],
)
def test_legacy_frontend_type_translates(legacy_type, expected_bank, expected_fusion):
    with pytest.warns(DeprecationWarning, match=legacy_type):
        cfg = CRLConfig(frontend_type=legacy_type)
    assert cfg.frontend_bank == expected_bank
    assert cfg.frontend_fusion == expected_fusion


def test_legacy_frontend_type_multiscale_no_warning():
    # Default frontend_type matches default (bank, fusion); no warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = CRLConfig(frontend_type="multiscale")
    assert cfg.frontend_bank == "multiscale"
    assert cfg.frontend_fusion == "early"


def test_legacy_morlet_per_sensor_params_promoted():
    """Setting frontend_type=morlet_per_sensor with morlet_per_sensor_params
    populates frontend_per_sensor_params."""
    with pytest.warns(DeprecationWarning):
        cfg = CRLConfig(
            frontend_type="morlet_per_sensor",
            morlet_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 8000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
            },
        )
    assert cfg.frontend_per_sensor_params["audio"]["freq_min"] == 20.0
    assert cfg.frontend_per_sensor_params["seismic"]["freq_max"] == 40.0


def test_new_schema_reverse_maps_to_frontend_type():
    """Setting frontend_bank/frontend_fusion populates frontend_type for back-compat."""
    cfg = CRLConfig(
        frontend_bank="morlet",
        frontend_fusion="late",
        frontend_per_sensor_params={
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "w0": 6.0,
                "target_tokens": 32,
                "receptive_cycles": 3.0,
            },
            "seismic": {
                "freq_min": 2.0,
                "freq_max": 40.0,
                "w0": 6.0,
                "target_tokens": 32,
                "receptive_cycles": 3.0,
            },
        },
    )
    assert cfg.frontend_type == "morlet_per_sensor"


def test_inconsistent_legacy_and_new_raises():
    # When BOTH legacy and new schema are set non-default but disagree, raise.
    # Note: if the new schema is at its default (multiscale, early), translation
    # treats it as unset and promotes the legacy field instead — that's the
    # reverse-promotion path, not an inconsistency.
    with pytest.raises(ValueError, match="Inconsistent frontend config"):
        CRLConfig(
            frontend_type="morlet_per_sensor",  # → (morlet, late)
            frontend_bank="morlet_learnable",  # disagrees on bank
            frontend_fusion="late",
            frontend_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 8000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                },
            },
        )


def test_morlet_freq_max_above_nyquist_raises():
    with pytest.raises(ValueError, match="Nyquist"):
        CRLConfig(
            frontend_bank="morlet",
            frontend_fusion="late",
            frontend_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 9000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                },
            },
            audio_target_rate=4000,  # Nyquist=2000, freq_max=9000 fails
        )


def test_required_keys_missing_multiscale_raises():
    with pytest.raises(ValueError, match="kernel_sizes"):
        CRLConfig(
            frontend_per_sensor_params={
                "audio": {"target_tokens": 32},  # missing kernel_sizes
                "seismic": {"target_tokens": 32, "kernel_sizes": [9]},
            },
        )


def test_required_keys_missing_morlet_raises():
    with pytest.raises(ValueError, match="freq_min"):
        CRLConfig(
            frontend_bank="morlet",
            frontend_fusion="late",
            frontend_per_sensor_params={
                "audio": {"target_tokens": 32},  # missing freq_min etc.
                "seismic": {"target_tokens": 32},
            },
        )


def test_required_keys_missing_target_tokens_raises():
    with pytest.raises(ValueError, match="target_tokens"):
        CRLConfig(
            frontend_per_sensor_params={
                "audio": {"kernel_sizes": [9, 19]},  # missing target_tokens
                "seismic": {"target_tokens": 32, "kernel_sizes": [9]},
            },
        )


def test_strides_length_mismatch_raises():
    with pytest.raises(ValueError, match="strides length"):
        CRLConfig(
            frontend_per_sensor_params={
                "audio": {
                    "target_tokens": 32,
                    "kernel_sizes": [9, 19, 39],
                    "strides": [4, 4],
                },  # length mismatch
                "seismic": {"target_tokens": 32, "kernel_sizes": [9]},
            },
        )
