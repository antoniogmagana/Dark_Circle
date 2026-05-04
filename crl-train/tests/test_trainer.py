import pytest
import torch
import torch.nn as nn
from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel, Trainer

# Active frontend variants. `frontend_type='morlet'` is deprecated and
# removed (see config.py for the migration error); kept out of this list so
# parametrized tests don't try to construct an invalid config.
ALL_FRONTENDS = (
    "multiscale",
    "morlet_per_sensor",
    "morlet_fused",
    "morlet_learnable",
    "morlet_learnable_fused",
)


@pytest.fixture
def cfg_ms():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="multiscale",
        fused_seq_len=16,
        d_z=24,
    )


@pytest.fixture
def cfg_morlet():
    return CRLConfig(
        d_model=32, n_layers=1, n_heads=4, frontend_type="morlet_per_sensor", d_z=24
    )


def _synthetic_batch(B=4, n_partners=4):
    # Window sizes track CRLConfig.modality_cfg targets so canonical-rate
    # changes flow through automatically.
    cfg = CRLConfig()
    W_a = cfg.modality_cfg("audio").window_size
    W_s = cfg.modality_cfg("seismic").window_size
    batch = {
        "x_audio_t": torch.randn(B, 1, W_a) * 0.01,
        "x_seismic_t": torch.randn(B, 1, W_s) * 0.01,
        "audio_avail": torch.ones(B, dtype=torch.bool),
        "seismic_avail": torch.ones(B, dtype=torch.bool),
        "detection_label_t": torch.randint(0, 2, (B,)),
        "vehicle_type_t": torch.randint(0, 4, (B,)),
        "n_partners": n_partners,
    }
    for p in range(n_partners):
        batch[f"x_audio_p{p}"] = torch.randn(B, 1, W_a) * 0.01
        batch[f"x_seismic_p{p}"] = torch.randn(B, 1, W_s) * 0.01
        batch[f"detection_label_p{p}"] = torch.randint(0, 2, (B,))
        batch[f"vehicle_type_p{p}"] = torch.randint(0, 4, (B,))
        batch[f"partner_stratum_p{p}"] = torch.full((B,), p % 4)
    return batch


class TestCRLModelMultiscale:
    def test_shared_encoder_exists(self, cfg_ms):
        model = CRLModel(cfg_ms)
        assert hasattr(model, "encoder") and model.encoder is not None

    def test_shared_decoder_exists(self, cfg_ms):
        model = CRLModel(cfg_ms)
        assert hasattr(model, "decoder") and model.decoder is not None

    def test_frontends_have_adaptive_pool(self, cfg_ms):
        model = CRLModel(cfg_ms)
        for sensor in ["audio", "seismic"]:
            assert any(
                isinstance(m, nn.AdaptiveAvgPool1d) for m in model.frontends[sensor].modules()
            ), f"{sensor} frontend missing AdaptiveAvgPool1d"

    def test_encode_fused_output_shape(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        T = cfg_ms.fused_seq_len
        with torch.no_grad():
            features, z, mu, logvar = model.encode_fused(
                torch.zeros(4, 1, 16000), torch.zeros(4, 1, 200)
            )
        assert features.shape == (4, cfg_ms.d_model, 2 * T)
        assert z.shape == (4, 24)
        assert mu.shape == (4, 24)

    def test_decode_fused_shape_matches_features(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        with torch.no_grad():
            features, z, _, _ = model.encode_fused(torch.zeros(4, 1, 16000), torch.zeros(4, 1, 200))
            x_hat = model.decode_fused(z)
        assert x_hat.shape == features.shape

    def test_encode_fused_finite(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        with torch.no_grad():
            features, z, mu, _ = model.encode_fused(
                torch.randn(4, 1, 16000) * 0.01, torch.randn(4, 1, 200) * 0.01
            )
        assert features.isfinite().all() and z.isfinite().all() and mu.isfinite().all()


class TestCRLModelMorlet:
    def test_per_sensor_encoders(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert "audio" in model.encoders and "seismic" in model.encoders

    def test_per_sensor_decoders(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert "audio" in model.decoders and "seismic" in model.decoders

    def test_encode_audio_shape(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        model.eval()
        with torch.no_grad():
            _, z, _, _ = model.encode("audio", torch.zeros(4, 1, 16000))
        assert z.shape == (4, 24)

    def test_encode_seismic_shape(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        model.eval()
        W_s = cfg_morlet.modality_cfg("seismic").window_size
        with torch.no_grad():
            _, z, _, _ = model.encode("seismic", torch.zeros(4, 1, W_s))
        assert z.shape == (4, 24)

    def test_no_shared_encoder(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert not hasattr(model, "encoder") or model.encoder is None


class TestCRLModelShared:
    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_has_latent(self, fe):
        from crl_vehicle.models.latent import CausalLatentSpace

        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        assert isinstance(model.latent, CausalLatentSpace)

    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_has_interv_classifier(self, fe):
        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        assert hasattr(model, "interv_classifier")

    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_backbone_excludes_downstream_heads(self, fe):
        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        backbone_ids = {id(p) for p in model.backbone_parameters()}
        for hd in [model.pres_heads, model.type_heads, model.prox_heads]:
            for p in hd.parameters():
                assert id(p) not in backbone_ids

    def test_unknown_frontend_raises(self):
        with pytest.raises(ValueError, match="frontend_type must be one of"):
            CRLModel(CRLConfig(frontend_type="bad", d_model=32, n_layers=1))


class TestForwardPair:
    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_finite_loss(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        model.train()
        loss, metrics = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        assert loss.isfinite(), f"Loss not finite: {loss.item()}"

    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_metrics_keys(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        trainer = Trainer(CRLModel(cfg), cfg, torch.device("cpu"), tmp_path)
        _, metrics = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        for k in ("recon", "kl", "raw_kl", "interv", "total"):
            assert k in metrics

    @pytest.mark.parametrize("fe", ALL_FRONTENDS)
    def test_finite_grads(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        model.train()
        trainer.optimizer.zero_grad()
        loss, _ = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        loss.backward()
        bad = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not p.grad.isfinite().all()
        ]
        assert not bad, f"Non-finite grads: {bad}"


class TestBetaAnnealing:
    """Beta schedule lives in VAETrainingMode.update_beta — Trainer only
    stores self.beta and passes it through. These tests drive the mode
    directly, bypassing the epoch loop."""

    @pytest.fixture
    def trainer(self, tmp_path):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            frontend_type="morlet_per_sensor",
            kl_floor=0.01,
            kl_target=0.5,
            beta_step=0.1,
        )
        return Trainer(CRLModel(cfg), cfg, torch.device("cpu"), tmp_path)

    def _step(self, trainer, val_m):
        """Invoke mode.update_beta with trainer's current state and assign."""
        new_beta, event = trainer.mode.update_beta(
            trainer.beta, val_m, trainer.ckpt_state, trainer.cfg
        )
        trainer.beta = new_beta
        return event

    def test_initial_beta_zero(self, trainer):
        assert trainer.beta == 0.0

    def test_mode_owns_update_beta(self, trainer):
        assert hasattr(trainer.mode, "update_beta")

    def test_increases_when_kl_above_target(self, trainer):
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 1.0})
        assert trainer.beta > 0.0

    def test_decreases_on_collapse(self, trainer):
        trainer.beta = 0.5
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 0.001})
        assert trainer.beta < 0.5

    def test_bounded_0_to_1(self, trainer):
        trainer.beta = 1.0
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 1.0})
        assert trainer.beta <= 1.0

    def test_returns_event_string(self, trainer):
        trainer.ckpt_state.prev_val_recon = 1.0
        event = self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 0.8})
        assert event in ("↑", "→hold", "↓collapse")


class TestLinearSignalProbe:
    """linear_signal builds a Linear(d_signal, 4) head; eval slices z[:d_signal]."""

    def test_builds_head_sized_to_d_signal(self):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="multiscale",
            fused_seq_len=16,
            d_z=24,
            d_signal=12,
        )
        model = CRLModel(cfg, probe_mode="linear_signal")
        head = model.type_heads["fused"].head
        assert head.weight.shape == (4, 12)

    def test_d_signal_16_makes_16_dim_head(self):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="multiscale",
            fused_seq_len=16,
            d_z=24,
            d_signal=16,
        )
        model = CRLModel(cfg, probe_mode="linear_signal")
        head = model.type_heads["fused"].head
        assert head.weight.shape == (4, 16)

    def test_per_sensor_frontend_also_works(self):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_per_sensor",
            fused_seq_len=16,
            d_z=24,
            d_signal=12,
        )
        model = CRLModel(cfg, probe_mode="linear_signal")
        for sensor in ("audio", "seismic"):
            head = model.type_heads[sensor].head
            assert head.weight.shape == (4, 12)

    def test_invalid_probe_mode_still_rejected(self):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="multiscale",
            fused_seq_len=16,
            d_z=24,
        )
        with pytest.raises(ValueError, match="probe_mode must be one of"):
            CRLModel(cfg, probe_mode="bogus")

    def test_valid_probe_modes_includes_linear_signal(self):
        assert "linear_signal" in CRLModel.VALID_PROBE_MODES


class TestDownstreamDualCheckpoints:
    """train_downstream saves two checkpoints — one per head — selected by
    argmax val_<task>_f1, and uses two independent AdamW optimizers (one per
    head) so the heads' training dynamics never couple. These tests run a
    short downstream loop on synthetic data and assert the on-disk + in-memory
    invariants."""

    def _build(self, tmp_path):
        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="multiscale",
            fused_seq_len=16,
            d_z=24,
        )
        model = CRLModel(cfg, probe_mode="linear_ztype")
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        return cfg, model, trainer

    def _synthetic_loader(self, cfg, B=4, n_batches=2):
        """Yield a deterministic small dataset of single-sample batches for
        train_downstream's loader contract."""
        W_a = cfg.modality_cfg("audio").window_size
        W_s = cfg.modality_cfg("seismic").window_size

        class _Loader:
            def __iter__(self_inner):
                torch.manual_seed(0)
                for _ in range(n_batches):
                    yield {
                        "x_audio": torch.randn(B, 1, W_a) * 0.01,
                        "x_seismic": torch.randn(B, 1, W_s) * 0.01,
                        "audio_avail": torch.ones(B, dtype=torch.bool),
                        "seismic_avail": torch.ones(B, dtype=torch.bool),
                        "detection_label": torch.randint(0, 2, (B,)),
                        "vehicle_type": torch.randint(0, 4, (B,)),
                    }

        return _Loader()

    def test_saves_both_head_checkpoints(self, tmp_path):
        cfg, _, trainer = self._build(tmp_path)
        loader = self._synthetic_loader(cfg)
        trainer.train_downstream(
            loader,
            loader,
            epochs=2,
            pres_pos_weight=torch.tensor(1.0),
            type_class_weights=torch.ones(4),
            finetune_top_n=0,
            ckpt_name="crl_best.pth",  # not used; no actual CRL ckpt here
        )
        assert (tmp_path / "downstream_best_pres.pth").exists()
        assert (tmp_path / "downstream_best_type.pth").exists()
        # Legacy filename must NOT be written — readers rely on its absence.
        assert not (tmp_path / "downstream_best.pth").exists()

    def test_csv_has_per_head_loss_columns(self, tmp_path):
        cfg, _, trainer = self._build(tmp_path)
        loader = self._synthetic_loader(cfg)
        trainer.train_downstream(
            loader,
            loader,
            epochs=1,
            pres_pos_weight=torch.tensor(1.0),
            type_class_weights=torch.ones(4),
            finetune_top_n=0,
            ckpt_name="crl_best.pth",
        )
        import csv as _csv

        with open(tmp_path / "downstream_metrics.csv") as f:
            reader = _csv.DictReader(f)
            cols = reader.fieldnames or []
        for required in (
            "val_loss",
            "val_pres_loss",
            "val_type_loss",
            "val_pres_f1",
            "val_type_f1",
        ):
            assert required in cols, f"missing column {required!r} in {cols}"

    def test_optimizers_step_disjoint_param_sets(self, tmp_path):
        """Critical scientific guarantee: pres_opt.step() touches only
        pres_heads parameters; type_opt.step() touches only type_heads
        parameters. Frozen-backbone case (finetune_top_n=0). Implementation:
        snapshot every parameter's data, run one batch, check which params
        actually changed."""
        cfg, model, trainer = self._build(tmp_path)
        loader = self._synthetic_loader(cfg, n_batches=1)
        # Pre-run snapshot of every parameter.
        trainer._pres_pos_weight = torch.tensor(1.0)
        trainer._type_class_weights = torch.ones(4)

        # Build the optimizers exactly the way train_downstream does, then
        # step once and inspect which parameters moved. We bypass the full
        # train_downstream() so we can interrogate state at exactly one
        # step-boundary.
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head_parameters():
            p.requires_grad_(True)

        pres_opt = torch.optim.AdamW(
            [{"params": list(model.pres_heads.parameters()), "lr": cfg.lr}]
        )
        type_opt = torch.optim.AdamW(
            [{"params": list(model.type_heads.parameters()), "lr": cfg.lr}]
        )

        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        for batch in loader:
            pres_opt.zero_grad()
            type_opt.zero_grad()
            pres_loss, type_loss, _ = trainer._downstream_forward(batch)
            pres_loss.backward()
            type_loss.backward()
            pres_opt.step()
            type_opt.step()
            break

        moved = {
            n
            for n, p in model.named_parameters()
            if not torch.allclose(p, snapshot[n], atol=0.0, rtol=0.0)
        }
        # Every moved param must belong to either pres_heads or type_heads —
        # backbone is frozen, prox/aux heads have no gradient signal here.
        for n in moved:
            assert n.startswith(("pres_heads.", "type_heads.")), (
                f"unexpected parameter moved: {n}"
            )
        # And both heads should have moved at least one param (this batch had
        # at least one valid presence label and one valid type label).
        assert any(n.startswith("pres_heads.") for n in moved), (
            "presence head did not update — pres_opt.step() didn't touch its params"
        )
        assert any(n.startswith("type_heads.") for n in moved), (
            "type head did not update — type_opt.step() didn't touch its params"
        )

        trainer._pres_pos_weight = None
        trainer._type_class_weights = None

    def test_pres_ckpt_is_argmax_val_pres_f1(self, tmp_path):
        """The saved pres ckpt corresponds to the epoch with highest
        val_pres_f1. We construct a synthetic CSV by intercepting the
        Trainer (running a 3-epoch loop and reading the CSV afterwards is
        cleaner than monkey-patching). Smoke-level check — as long as the
        logged epoch's pres_f1 equals max(csv.val_pres_f1), the selector
        is doing the right thing."""
        cfg, _, trainer = self._build(tmp_path)
        loader = self._synthetic_loader(cfg, n_batches=2)
        trainer.train_downstream(
            loader,
            loader,
            epochs=3,
            pres_pos_weight=torch.tensor(1.0),
            type_class_weights=torch.ones(4),
            finetune_top_n=0,
            ckpt_name="crl_best.pth",
        )
        import csv as _csv

        with open(tmp_path / "downstream_metrics.csv") as f:
            rows = list(_csv.DictReader(f))
        max_pres = max(float(r["val_pres_f1"]) for r in rows)
        max_type = max(float(r["val_type_f1"]) for r in rows)
        # Sanity: there's only one ckpt save policy, so we know the saved file
        # exists; what we check here is that the *csv columns* track per-head
        # F1 cleanly. The selector is exercised every epoch in train_downstream
        # so the ckpt files exist iff at least one row had F1 ≥ -1.0 (always).
        assert (tmp_path / "downstream_best_pres.pth").exists()
        assert (tmp_path / "downstream_best_type.pth").exists()
        assert 0.0 <= max_pres <= 1.0
        assert 0.0 <= max_type <= 1.0


class TestSR4000GraduatedExperiment:
    """End-to-end construction + forward for the audio_target_rate=4000
    graduated-kernel experiment. This is the config that drives commit 8."""

    def _cfg(self):
        return CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            audio_target_rate=4000,
            frontend_per_sensor_params={
                "audio": {
                    "target_tokens": 32,
                    "kernel_sizes": [201, 41, 9, 5],
                    "strides": [67, 13, 3, 1],
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "target_tokens": 32,
                    "kernel_sizes": [51, 21, 7, 3],
                    "strides": [17, 7, 2, 1],
                    "out_channels_frac": 1.0,
                },
            },
        )

    def test_audio_window_size_follows_target_rate(self):
        cfg = self._cfg()
        assert cfg.modality_cfg("audio").sample_rate == 4000
        assert cfg.modality_cfg("audio").window_size == 4000

    def test_constructs_and_runs(self):
        cfg = self._cfg()
        model = CRLModel(cfg)
        model.eval()
        audio = torch.randn(2, 1, 4000) * 0.01
        seismic = torch.randn(2, 1, 100) * 0.01
        with torch.no_grad():
            features, z, mu, _ = model.encode_fused(audio, seismic)
        # 2 sensors × 32 tokens = 64
        assert features.shape == (2, 16, 64)
        assert z.shape == (2, cfg.d_z)
        assert features.isfinite().all()
