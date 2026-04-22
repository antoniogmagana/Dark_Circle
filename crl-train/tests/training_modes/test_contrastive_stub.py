"""Interface load-test: does a minimal contrastive TrainingMode survive the
Trainer loop, the CSV writer, early-stop logic, and checkpoint save paths?

This is a throwaway stub — it does not implement real contrastive learning.
It only demonstrates that the Checkpoint 1 abstractions do not leak VAE-
specific assumptions into the Trainer. If this test crashes on a missing
`val_recon` or `val_ref_elbo`, we have a leak and Checkpoint 3 will have to
edit Trainer. If it passes, the interfaces are Checkpoint-3-ready.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crl_vehicle.config import CRLConfig
from crl_vehicle.training_modes import CheckpointState, TrainingMode


class _ContrastiveStub(TrainingMode):
    """Minimal non-VAE stub: encode anchor & partner, return dot-product loss.

    No decoder, no KL, no aux losses, no intervention — just proves the
    Trainer loop tolerates a mode that reports a different metric schema.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward_pair(self, model, batch, beta, device):
        # Trivial "loss": mean-squared latent of the anchor. Not meaningful,
        # just differentiable so Trainer's backward() succeeds.
        avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
        if not avail.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        x_a = batch["x_audio_t"][avail].to(device)
        x_s = batch["x_seismic_t"][avail].to(device)
        if self.config.frontend_type == "multiscale":
            _, z, _, _ = model.encode_fused(x_a, x_s)
        else:
            _, z, _, _ = model.encode(model.sensors[0], x_a)
        loss = z.pow(2).mean()
        return loss, {"contrastive_loss": loss.item()}

    def val_metrics_summary(self, val_m: dict) -> dict:
        # No derived metric — contrastive has no ELBO to compute.
        return val_m

    def update_beta(self, beta, val_m, state, config):
        # Contrastive has no beta schedule.
        return (0.0, "→hold")

    def should_save_checkpoint(self, val_m, epoch, state):
        # Stub: save on every epoch, track best by contrastive_loss (lower = better).
        loss = val_m.get("val_contrastive_loss", float("inf"))
        best = state.bests.get("val_contrastive_loss", float("inf"))
        save = loss < best - 1e-5
        if save:
            state.bests["val_contrastive_loss"] = loss
            state.best_epochs["val_contrastive_loss"] = epoch
            state.patience_count = 0
        else:
            state.patience_count += 1
        return {"contrastive_best.pth": save}

    def early_stop_metric(self) -> str:
        return "val_contrastive_loss"

    def early_stop_mode(self) -> str:
        return "min"


def _dummy_loader(n_batches: int = 2, B: int = 2):
    class _L:
        def __iter__(self):
            for _ in range(n_batches):
                yield {
                    "x_audio_t":         torch.randn(B, 1, 16000) * 0.01,
                    "x_seismic_t":       torch.randn(B, 1, 200) * 0.01,
                    "audio_avail":       torch.ones(B, dtype=torch.bool),
                    "seismic_avail":     torch.ones(B, dtype=torch.bool),
                    "detection_label_t": torch.zeros(B, dtype=torch.long),
                    "vehicle_type_t":    torch.zeros(B, dtype=torch.long),
                    "n_partners":        0,
                }
    return _L()


def test_trainer_survives_non_vae_mode(tmp_path, monkeypatch):
    """Swap the default mode factory to a contrastive stub and run two epochs."""
    from crl_vehicle import training_modes as tm
    from training import trainer as trainer_mod

    cfg = CRLConfig(
        d_model=32, n_layers=1, frontend_type="multiscale",
        fused_seq_len=16, d_z=24, n_epochs=2, early_stop_patience=10,
    )

    # Monkey-patch the factory so Trainer receives the stub.
    stub = _ContrastiveStub(cfg)
    monkeypatch.setattr(trainer_mod, "build_training_mode", lambda c: stub)

    model = trainer_mod.CRLModel(cfg)
    trainer = trainer_mod.Trainer(model, cfg, torch.device("cpu"), tmp_path)

    # train_crl must not raise even though val_m has no val_recon, val_raw_kl,
    # val_ref_elbo, or val_aux_type_f1.
    trainer.train_crl(
        train_loader=_dummy_loader(n_batches=2),
        val_loader=_dummy_loader(n_batches=1),
        epochs=2,
    )

    # Checkpoint file should exist.
    assert (tmp_path / "contrastive_best.pth").exists()
    assert (tmp_path / "crl_final.pth").exists()
    # CSV should have been written.
    assert (tmp_path / "crl_metrics.csv").exists()
    # Summary JSON should contain whatever the stub reported.
    import json
    summary = json.loads((tmp_path / "crl_checkpoint_summary.json").read_text())
    # Stub uses the generic base checkpoint_summary() — should have bests & best_epochs.
    assert "bests" in summary
    assert "best_epochs" in summary
