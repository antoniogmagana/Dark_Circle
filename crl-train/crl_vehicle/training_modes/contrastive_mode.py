from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.data.dataset import (
    STRATUM_CONSEC, STRATUM_CROSS_DS, STRATUM_DIFF_TYPE, STRATUM_SAME_TYPE,
)
from crl_vehicle.losses.contrastive import nt_xent_loss
from crl_vehicle.training_modes.base import CheckpointState, TrainingMode


_POSITIVE_STRATA = (STRATUM_CONSEC, STRATUM_SAME_TYPE)


class ContrastiveTrainingMode(TrainingMode):
    """NT-Xent contrastive pre-training over StratifiedPairDataset partners.

    Positives per anchor:  partners with stratum in {CONSEC, SAME_TYPE}.
    Negatives per anchor:  remaining partners (DIFF_TYPE, CROSS_DS) for this
                           anchor, plus all partners of other anchors in batch.

    Architecture: encoder(s) produce `mu` (posterior mean — used as the
    deterministic representation, no sampling); a small projection head
    (d_z → d_proj → d_proj) maps to contrastive space, L2-normalized.
    Projection head lives on the mode so CRLModel stays VAE-shaped; its
    params flow through the optimizer via Trainer's mode-params group.

    No decoder, no KL, no aux losses during CRL. Downstream probes are
    trained post-hoc by `train_downstream` on the frozen encoder.
    """

    CKPT_BEST = "crl_best.pth"

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        d_z    = config.d_z
        d_proj = config.contrastive_d_proj
        self.projection = nn.Sequential(
            nn.Linear(d_z, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )
        self.temperature = config.contrastive_temperature

    def forward_pair(
        self,
        model: nn.Module,
        batch: dict,
        beta: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        n_partners = batch["n_partners"]
        if n_partners == 0:
            return torch.zeros((), device=device, requires_grad=True), \
                   {"contrastive_loss": 0.0}

        if model.is_fused_frontend():
            mu_t, mu_parts, is_pos = self._encode_fused(model, batch, n_partners, device)
        else:
            mu_t, mu_parts, is_pos = self._encode_per_sensor(model, batch, n_partners, device)

        if mu_t is None:
            return torch.zeros((), device=device, requires_grad=True), \
                   {"contrastive_loss": 0.0}

        z_a = F.normalize(self.projection(mu_t), dim=-1)
        B, P, _ = mu_parts.shape
        z_p = F.normalize(
            self.projection(mu_parts.reshape(B * P, -1)).reshape(B, P, -1),
            dim=-1,
        )
        loss = nt_xent_loss(z_a, z_p, is_pos, temperature=self.temperature)
        return loss, {"contrastive_loss": loss.item()}

    def _encode_fused(self, model, batch, n_partners, dev):
        avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
        if not avail.any():
            return None, None, None
        x_a = batch["x_audio_t"][avail].to(dev)
        x_s = batch["x_seismic_t"][avail].to(dev)
        _, _, mu_t, _ = model.encode_fused(x_a, x_s)

        mu_parts = []
        strata   = []
        for p in range(n_partners):
            xa = batch[f"x_audio_p{p}"][avail].to(dev)
            xs = batch[f"x_seismic_p{p}"][avail].to(dev)
            _, _, mu_p, _ = model.encode_fused(xa, xs)
            mu_parts.append(mu_p)
            strata.append(batch[f"partner_stratum_p{p}"][avail].to(dev))
        mu_parts = torch.stack(mu_parts, dim=1)     # (B, P, d_z)
        strata_t = torch.stack(strata, dim=1)        # (B, P)
        is_pos = self._positive_mask(strata_t)
        return mu_t, mu_parts, is_pos

    def _encode_per_sensor(self, model, batch, n_partners, dev):
        """Mean over available sensors — gives one anchor representation per
        sample even when only one modality is present."""
        mu_a_list: list[torch.Tensor] = []
        mu_p_list: list[torch.Tensor] = []
        strata_list: list[torch.Tensor] = []

        # Build a per-sample mask of samples where at least one sensor is
        # available, and encode each available sensor.
        any_avail = torch.zeros(batch["audio_avail"].shape[0], dtype=torch.bool)
        per_sensor_mu_a: dict[str, torch.Tensor] = {}
        per_sensor_mu_p: dict[str, list[torch.Tensor]] = {}

        for sensor in model.sensors:
            avail = batch[f"{sensor}_avail"].bool()
            if not avail.any():
                continue
            any_avail = any_avail | avail
            x = batch[f"x_{sensor}_t"][avail].to(dev)
            _, _, mu_t, _ = model.encode(sensor, x)
            per_sensor_mu_a[sensor] = (avail, mu_t)

            partner_mus = []
            for p in range(n_partners):
                xp = batch[f"x_{sensor}_p{p}"][avail].to(dev)
                _, _, mu_p, _ = model.encode(sensor, xp)
                partner_mus.append(mu_p)
            per_sensor_mu_p[sensor] = partner_mus

        if not any_avail.any():
            return None, None, None

        # Average across sensors for each sample present in `any_avail`.
        N = int(any_avail.sum().item())
        mu_t_sum = torch.zeros(N, self.config.d_z, device=dev)
        mu_t_cnt = torch.zeros(N, 1, device=dev)
        mu_p_sum = torch.zeros(N, n_partners, self.config.d_z, device=dev)
        mu_p_cnt = torch.zeros(N, 1, device=dev)

        # Map original indices → compressed index under any_avail.
        remap = -torch.ones(any_avail.shape[0], dtype=torch.long, device=dev)
        remap[any_avail] = torch.arange(N, device=dev)

        for sensor, (avail, mu_t) in per_sensor_mu_a.items():
            idx = remap[avail]
            mu_t_sum[idx] += mu_t
            mu_t_cnt[idx] += 1
            for p, mu_p in enumerate(per_sensor_mu_p[sensor]):
                mu_p_sum[idx, p] += mu_p
            mu_p_cnt[idx] += 1  # counts only depend on sample, not p

        mu_t    = mu_t_sum / mu_t_cnt.clamp(min=1)
        mu_parts = mu_p_sum / mu_p_cnt.clamp(min=1).unsqueeze(-1)

        # Strata are per-sample (not per-sensor). Use the mask of samples with
        # at least one sensor.
        strata = torch.stack(
            [batch[f"partner_stratum_p{p}"][any_avail].to(dev)
             for p in range(n_partners)],
            dim=1,
        )
        is_pos = self._positive_mask(strata)
        return mu_t, mu_parts, is_pos

    @staticmethod
    def _positive_mask(strata: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(strata, dtype=torch.bool)
        for s in _POSITIVE_STRATA:
            out = out | (strata == s)
        return out

    def val_metrics_summary(self, val_m: dict) -> dict:
        return val_m

    def update_beta(self, beta, val_m, state, config):
        return (0.0, "→hold")

    def should_save_checkpoint(
        self, val_m: dict, epoch: int, state: CheckpointState
    ) -> dict[str, bool]:
        loss = val_m.get("val_contrastive_loss", float("inf"))
        best = state.bests.get("val_contrastive_loss", float("inf"))
        save = loss < best - 1e-5
        if save:
            state.bests["val_contrastive_loss"] = loss
            state.best_epochs["val_contrastive_loss"] = epoch
            state.patience_count = 0
        else:
            state.patience_count += 1
        return {self.CKPT_BEST: save}

    def early_stop_metric(self) -> str:
        return "val_contrastive_loss"

    def early_stop_mode(self) -> str:
        return "min"

    def checkpoint_summary(self, state: CheckpointState) -> dict:
        return {
            "best_contrastive_loss": round(
                state.bests.get("val_contrastive_loss", float("inf")), 6
            ),
            "best_epoch": state.best_epochs.get("val_contrastive_loss", -1),
            "checkpoints": {
                self.CKPT_BEST: "selected by val_contrastive_loss (lower = better)",
                "crl_final.pth": "last epoch (may be post-early-stop)",
            },
        }
