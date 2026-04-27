from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.data.dataset import STRATUM_CONSEC
from crl_vehicle.data.transforms import apply_intervention_batch
from crl_vehicle.losses.crl_loss import kl_divergence, reconstruction_loss
from crl_vehicle.losses.disentanglement import (
    cross_modal_alignment_loss,
    intervention_invariance_loss,
    temporal_stability_loss,
)
from crl_vehicle.models.heads import LinearPresenceHead, LinearTypeHead
from crl_vehicle.models.latent import SplitLatentSpace
from crl_vehicle.training_modes.base import CheckpointState, TrainingMode


def _empty_aux_metrics() -> dict:
    empty_f = torch.empty(0)
    empty_l = torch.empty(0, dtype=torch.long)
    return {
        "recon": 0.0, "kl": 0.0, "raw_kl": 0.0,
        "align": 0.0, "stab": 0.0, "interv_inv": 0.0,
        "total": 0.0,
        "aux_pres_logits": empty_f, "aux_pres_labels": empty_l,
        "aux_type_logits": empty_f, "aux_type_labels": empty_l,
    }


class DisentangledVAETrainingMode(TrainingMode):
    """VAE with z = z_signal ∪ z_env, disentangled by physics-derived losses.

    Two-block latent (SplitLatentSpace): the first d_signal dims are claimed
    as vehicle-relevant, the rest as environment/noise. Routing is enforced
    by three losses, not by per-feature dim assignment.

    Three additional losses beyond standard ELBO:
      - cross_modal_alignment(mu_signal_audio, mu_signal_seismic)
        — same source excites both sensors; their signal estimates must agree.
        Per-sensor frontends only; fused frontends skip (no separate per-modality mu).
      - temporal_stability(mu_env_t, mu_env_tn)
        — env varies slowly; consecutive-window pairs penalize env drift.
      - intervention_invariance(mu_signal_clean, mu_signal_intervened)
        — vehicle signal must be unchanged under noise interventions; pushes
        intervention-induced variation into z_env.

    Aux heads (presence + type) read the *full* z_signal — no per-feature
    sub-slicing within the block. This drops the D_PRES=4/D_TYPE=6 ceiling
    that capped the previous architecture's labeled subspace.

    No intervention classifier (replaced by invariance loss). Standard prior
    on the full z (KL term unchanged from VAE mode).

    Dual checkpoints (same as VAETrainingMode):
      - crl_best.pth          → best val_ref_elbo (recon + raw_kl)
      - crl_best_aux_type.pth → best val_aux_type_f1 (downstream-proxy)
    """

    CKPT_REF_ELBO    = "crl_best.pth"
    CKPT_AUX_TYPE_F1 = "crl_best_aux_type.pth"

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.latent = SplitLatentSpace(d_z=config.d_z, d_signal=config.d_signal)
        # Heads on the mode (not the model) so they're sized by d_signal,
        # not by the model's CausalLatentSpace.D_PRES/D_TYPE constants.
        # Trainer's mode-params optimizer group picks them up via mode.parameters().
        self.pres_head = LinearPresenceHead(d_in=config.d_signal)
        self.type_head = LinearTypeHead(d_in=config.d_signal, n_classes=4)
        self.lambda_align       = config.lambda_align
        self.lambda_stab        = config.lambda_stab
        self.lambda_interv_inv  = config.lambda_interv_inv

    # ------------------------------------------------------------------
    # Forward pair — dispatches by frontend topology.
    # ------------------------------------------------------------------

    def forward_pair(
        self,
        model: nn.Module,
        batch: dict,
        beta: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        if model.is_fused_frontend():
            return self._forward_pair_fused(model, batch, beta, device)
        return self._forward_pair_per_sensor(model, batch, beta, device)

    # ------------------------------------------------------------------
    # Fused (multiscale early-fusion): no separate per-modality mu, so no
    # cross-modal alignment loss. Stability + invariance still apply.
    # ------------------------------------------------------------------

    def _forward_pair_fused(
        self, model, batch, beta, dev
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
        if not avail.any():
            zero = torch.tensor(0.0, device=dev, requires_grad=True)
            return zero, _empty_aux_metrics()

        x_a = batch["x_audio_t"][avail].to(dev)
        x_s = batch["x_seismic_t"][avail].to(dev)

        features, z_t, mu_t, lv_t = model.encode_fused(x_a, x_s)
        x_hat = model.decode_fused(z_t)

        recon  = reconstruction_loss(x_hat, features.detach())
        raw_kl = kl_divergence(mu_t, lv_t, beta=1.0)
        kl     = beta * raw_kl

        mu_signal, mu_env = self.latent.split(mu_t)
        det_t  = batch["detection_label_t"][avail].float().to(dev)
        type_t = batch["vehicle_type_t"][avail].to(dev)

        aux_pres_logit = self.pres_head(mu_signal).squeeze(-1)
        aux_pres = F.binary_cross_entropy_with_logits(aux_pres_logit, det_t)

        valid_type = type_t >= 0
        aux_type = torch.tensor(0.0, device=dev)
        type_logit_valid  = torch.empty(0, device=dev)
        type_labels_valid = torch.empty(0, dtype=torch.long, device=dev)
        if valid_type.any():
            type_logit_valid  = self.type_head(mu_signal[valid_type])
            type_labels_valid = type_t[valid_type].long()
            aux_type = F.cross_entropy(type_logit_valid, type_labels_valid)

        # Temporal stability on env via consecutive partner.
        stab = torch.tensor(0.0, device=dev)
        n_partners = batch["n_partners"]
        if n_partners > 0:
            x_a_p0 = batch["x_audio_p0"][avail].to(dev)
            x_s_p0 = batch["x_seismic_p0"][avail].to(dev)
            _, _, mu_tn, _ = model.encode_fused(x_a_p0, x_s_p0)
            _, mu_env_tn = self.latent.split(mu_tn)
            strata_p0 = batch["partner_stratum_p0"][avail].to(dev)
            consec_mask = (strata_p0 == STRATUM_CONSEC)
            stab = temporal_stability_loss(mu_env, mu_env_tn, consec_mask)

        # Intervention invariance: re-encode anchor with random noise applied.
        # Sample rates come from config so they track _SOURCE_RATES targets.
        x_a_int = _apply_intervention_batch(
            x_a, sample_rate=cfg.modality_cfg("audio").sample_rate
        )
        x_s_int = _apply_intervention_batch(
            x_s, sample_rate=cfg.modality_cfg("seismic").sample_rate
        )
        _, _, mu_int, _ = model.encode_fused(x_a_int, x_s_int)
        mu_signal_int, _ = self.latent.split(mu_int)
        interv_inv = intervention_invariance_loss(mu_signal, mu_signal_int)

        total = (recon + kl
                 + cfg.lambda_aux_pres * aux_pres
                 + cfg.lambda_aux_type * aux_type
                 + self.lambda_stab        * stab
                 + self.lambda_interv_inv  * interv_inv)

        metrics = {
            "recon":      recon.item(),
            "kl":         kl.item(),
            "raw_kl":     raw_kl.item(),
            "align":      0.0,                      # n/a for fused frontends
            "stab":       stab.item(),
            "interv_inv": interv_inv.item(),
            "total":      total.item(),
            "aux_pres_logits": aux_pres_logit.detach().cpu(),
            "aux_pres_labels": det_t.detach().long().cpu(),
            "aux_type_logits": type_logit_valid.detach().cpu(),
            "aux_type_labels": type_labels_valid.detach().cpu(),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Per-sensor (morlet late-fusion): separate per-modality mu enables
    # the full loss recipe including cross-modal alignment.
    # ------------------------------------------------------------------

    def _forward_pair_per_sensor(
        self, model, batch, beta, dev
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        # Per-sensor sample rates from config so canonical-rate changes flow through.
        sample_rates = {s: cfg.modality_cfg(s).sample_rate for s in model.sensors}

        per_sensor: dict[str, dict] = {}
        for sensor in model.sensors:
            avail_cpu = batch[f"{sensor}_avail"].bool()
            if not avail_cpu.any():
                continue
            x = batch[f"x_{sensor}_t"][avail_cpu].to(dev)
            features, z_t, mu_t, lv_t = model.encode(sensor, x)
            x_hat = model.decode(sensor, z_t)
            mu_signal, mu_env = self.latent.split(mu_t)
            per_sensor[sensor] = {
                "avail":     avail_cpu.to(dev),  # device-resident for batch-level masking
                "avail_cpu": avail_cpu,           # original CPU mask for batch[...] indexing
                "x": x, "features": features,
                "z_t": z_t, "mu_t": mu_t, "lv_t": lv_t,
                "mu_signal": mu_signal, "mu_env": mu_env, "x_hat": x_hat,
            }

        if not per_sensor:
            zero = torch.tensor(0.0, device=dev, requires_grad=True)
            return zero, _empty_aux_metrics()

        det_t  = batch["detection_label_t"].float().to(dev)
        type_t = batch["vehicle_type_t"].to(dev)

        # Per-sensor recon + KL aggregated across sensors.
        recon_total  = torch.tensor(0.0, device=dev)
        kl_total     = torch.tensor(0.0, device=dev)
        raw_kl_total = torch.tensor(0.0, device=dev)
        for sensor, p in per_sensor.items():
            recon_total  = recon_total  + reconstruction_loss(p["x_hat"], p["features"].detach())
            raw_kl       = kl_divergence(p["mu_t"], p["lv_t"], beta=1.0)
            kl_total     = kl_total     + beta * raw_kl
            raw_kl_total = raw_kl_total + raw_kl
        n_active = len(per_sensor)
        recon_total  = recon_total  / n_active
        kl_total     = kl_total     / n_active
        raw_kl_total = raw_kl_total / n_active

        # Aux heads — average mu_signal across available sensors per sample.
        mu_signal_avg = _average_per_sensor(per_sensor, "mu_signal", dev, key_dim=cfg.d_signal)
        mu_env_avg    = _average_per_sensor(
            per_sensor, "mu_env", dev, key_dim=cfg.d_z - cfg.d_signal
        )
        any_avail = torch.zeros_like(det_t, dtype=torch.bool)
        for p in per_sensor.values():
            any_avail = any_avail | p["avail"]

        det_valid  = det_t[any_avail]
        type_valid = type_t[any_avail]

        aux_pres_logit = self.pres_head(mu_signal_avg).squeeze(-1)
        aux_pres = F.binary_cross_entropy_with_logits(aux_pres_logit, det_valid)

        valid_type_mask = type_valid >= 0
        aux_type = torch.tensor(0.0, device=dev)
        type_logit_valid  = torch.empty(0, device=dev)
        type_labels_valid = torch.empty(0, dtype=torch.long, device=dev)
        if valid_type_mask.any():
            type_logit_valid  = self.type_head(mu_signal_avg[valid_type_mask])
            type_labels_valid = type_valid[valid_type_mask].long()
            aux_type = F.cross_entropy(type_logit_valid, type_labels_valid)

        # Cross-modal alignment: only if BOTH audio and seismic are present
        # for at least one sample. Compare on the joint-availability subset.
        align = torch.tensor(0.0, device=dev)
        if "audio" in per_sensor and "seismic" in per_sensor:
            both = per_sensor["audio"]["avail"] & per_sensor["seismic"]["avail"]
            if both.any():
                # Need to subselect each sensor's mu_signal to the joint subset.
                a_idx = _remap_to_compressed(per_sensor["audio"]["avail"], both, dev)
                s_idx = _remap_to_compressed(per_sensor["seismic"]["avail"], both, dev)
                mu_a = per_sensor["audio"]["mu_signal"][a_idx]
                mu_s = per_sensor["seismic"]["mu_signal"][s_idx]
                align = cross_modal_alignment_loss(mu_a, mu_s)

        # Temporal stability on env via consecutive partner.
        stab = torch.tensor(0.0, device=dev)
        n_partners = batch["n_partners"]
        if n_partners > 0:
            stab_terms: list[torch.Tensor] = []
            for sensor, p in per_sensor.items():
                x_p0 = batch[f"x_{sensor}_p0"][p["avail_cpu"]].to(dev)
                _, _, mu_tn, _ = model.encode(sensor, x_p0)
                _, mu_env_tn = self.latent.split(mu_tn)
                strata_p0 = batch["partner_stratum_p0"][p["avail_cpu"]].to(dev)
                consec_mask = (strata_p0 == STRATUM_CONSEC)
                stab_terms.append(temporal_stability_loss(p["mu_env"], mu_env_tn, consec_mask))
            stab = torch.stack(stab_terms).mean() if stab_terms else stab

        # Intervention invariance: re-encode each sensor with intervention.
        interv_inv_terms: list[torch.Tensor] = []
        for sensor, p in per_sensor.items():
            x_int = _apply_intervention_batch(p["x"], sample_rate=sample_rates[sensor])
            _, _, mu_int, _ = model.encode(sensor, x_int)
            mu_signal_int, _ = self.latent.split(mu_int)
            interv_inv_terms.append(intervention_invariance_loss(p["mu_signal"], mu_signal_int))
        interv_inv = torch.stack(interv_inv_terms).mean() if interv_inv_terms \
                     else torch.tensor(0.0, device=dev)

        total = (recon_total + kl_total
                 + cfg.lambda_aux_pres * aux_pres
                 + cfg.lambda_aux_type * aux_type
                 + self.lambda_align       * align
                 + self.lambda_stab        * stab
                 + self.lambda_interv_inv  * interv_inv)

        metrics = {
            "recon":      recon_total.item(),
            "kl":         kl_total.item(),
            "raw_kl":     raw_kl_total.item(),
            "align":      align.item(),
            "stab":       stab.item(),
            "interv_inv": interv_inv.item(),
            "total":      total.item(),
            "aux_pres_logits": aux_pres_logit.detach().cpu(),
            "aux_pres_labels": det_valid.detach().long().cpu(),
            "aux_type_logits": type_logit_valid.detach().cpu(),
            "aux_type_labels": type_labels_valid.detach().cpu(),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Derived metrics, beta, checkpoint selection — same shape as VAE mode.
    # ------------------------------------------------------------------

    def val_metrics_summary(self, val_m: dict) -> dict:
        out = dict(val_m)
        out["val_ref_elbo"] = (
            val_m.get("val_recon", 0.0) + val_m.get("val_raw_kl", 0.0)
        )
        return out

    def update_beta(
        self, beta: float, val_m: dict, state: CheckpointState, config
    ) -> tuple[float, str]:
        raw_kl = val_m["val_raw_kl"]
        recon_improving = (
            val_m["val_recon"] < state.prev_val_recon - config.recon_min_delta
        )
        state.prev_val_recon = val_m["val_recon"]
        if raw_kl < config.kl_floor:
            return (max(0.0, beta - config.beta_step), "↓collapse")
        if recon_improving or raw_kl > config.kl_target:
            return (min(1.0, beta + config.beta_step), "↑")
        return (beta, "→hold")

    def should_save_checkpoint(
        self, val_m: dict, epoch: int, state: CheckpointState
    ) -> dict[str, bool]:
        save: dict[str, bool] = {}

        ref_elbo = val_m["val_ref_elbo"]
        best_ref = state.bests.get("val_ref_elbo", float("inf"))
        if ref_elbo < best_ref - 1e-5:
            state.bests["val_ref_elbo"] = ref_elbo
            state.best_epochs["val_ref_elbo"] = epoch
            state.patience_count = 0
            save[self.CKPT_REF_ELBO] = True
        else:
            state.patience_count += 1
            save[self.CKPT_REF_ELBO] = False

        aux_type_f1 = val_m.get("val_aux_type_f1", 0.0)
        best_aux = state.bests.get("val_aux_type_f1", -1.0)
        if aux_type_f1 > best_aux + 1e-5:
            state.bests["val_aux_type_f1"] = aux_type_f1
            state.best_epochs["val_aux_type_f1"] = epoch
            save[self.CKPT_AUX_TYPE_F1] = True
        else:
            save[self.CKPT_AUX_TYPE_F1] = False

        return save

    def early_stop_metric(self) -> str:
        return "val_ref_elbo"

    def early_stop_mode(self) -> str:
        return "min"

    def checkpoint_summary(self, state: CheckpointState) -> dict:
        return {
            "best_ref_elbo":       round(state.bests.get("val_ref_elbo", float("inf")), 6),
            "best_aux_type_f1":    round(state.bests.get("val_aux_type_f1", -1.0), 4),
            "best_aux_type_epoch": state.best_epochs.get("val_aux_type_f1", -1),
            "checkpoints": {
                self.CKPT_REF_ELBO:    "selected by val_ref_elbo (recon + raw_kl at beta=1)",
                self.CKPT_AUX_TYPE_F1: "selected by val_aux_type_f1 (downstream-proxy signal)",
                "crl_final.pth":       "last epoch (may be post-early-stop)",
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_intervention_batch(x: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply a random NON-noop intervention (1..N) to each sample in the batch.

    Thin wrapper over crl_vehicle.data.transforms.apply_intervention_batch that
    forces every sample to receive an intervention (id 0 = no-op excluded).
    Stays on the input device — no host↔device bounces.
    """
    from crl_vehicle.data.transforms import N_INTERVENTIONS
    B = x.shape[0]
    interv_ids = torch.randint(1, N_INTERVENTIONS + 1, (B,), device=x.device)
    return apply_intervention_batch(x.detach(), sample_rate, interv_ids=interv_ids)


def _remap_to_compressed(
    avail_mask: torch.Tensor, target_subset: torch.Tensor, dev: torch.device
) -> torch.Tensor:
    """Given an `avail_mask` (B,) bool over the full batch, and a `target_subset`
    (B,) bool that is a subset of `avail_mask`, return the indices into the
    avail_mask-compressed tensor that select rows corresponding to target_subset.
    """
    n_avail = int(avail_mask.sum().item())
    full_to_compressed = -torch.ones(avail_mask.shape[0], dtype=torch.long, device=dev)
    full_to_compressed[avail_mask] = torch.arange(n_avail, device=dev)
    return full_to_compressed[target_subset]


def _average_per_sensor(
    per_sensor: dict, key: str, dev: torch.device, key_dim: int
) -> torch.Tensor:
    """Average per-sample across whichever sensors are available.

    Returns shape (n_any_avail, key_dim) where n_any_avail is the count of
    samples with at least one sensor present.

    Expects p["avail"] to be on `dev` (set by _forward_pair_per_sensor).
    """
    B = next(iter(per_sensor.values()))["avail"].shape[0]
    any_avail = torch.zeros(B, dtype=torch.bool, device=dev)
    for p in per_sensor.values():
        any_avail = any_avail | p["avail"]
    N = int(any_avail.sum().item())

    full_to_compressed = -torch.ones(B, dtype=torch.long, device=dev)
    full_to_compressed[any_avail] = torch.arange(N, device=dev)

    sum_buf = torch.zeros(N, key_dim, device=dev)
    cnt_buf = torch.zeros(N, 1, device=dev)
    for p in per_sensor.values():
        idx = full_to_compressed[p["avail"]]
        sum_buf[idx] += p[key]
        cnt_buf[idx] += 1
    return sum_buf / cnt_buf.clamp(min=1)
