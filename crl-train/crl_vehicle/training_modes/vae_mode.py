from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crl_vehicle.losses.crl_loss import (
    intervention_matching_loss,
    kl_divergence,
    reconstruction_loss,
)
from crl_vehicle.models.intervention import label_change_target
from crl_vehicle.priors import Prior
from crl_vehicle.training_modes.base import CheckpointState, TrainingMode


def _empty_aux_metrics() -> dict:
    empty_f = torch.empty(0)
    empty_l = torch.empty(0, dtype=torch.long)
    return {
        "recon": 0.0, "kl": 0.0, "raw_kl": 0.0, "interv": 0.0, "total": 0.0,
        "aux_pres_logits": empty_f, "aux_pres_labels": empty_l,
        "aux_type_logits": empty_f, "aux_type_labels": empty_l,
    }


class VAETrainingMode(TrainingMode):
    """Classical CRL pre-training: recon + beta*KL + interv + aux losses.

    Owns:
      - dispatch between fused (multiscale early-fusion) and per-sensor
        (morlet late-fusion) forward paths, matching the existing CRLModel.
      - KL computation via an injected Prior (StandardPrior by default;
        Checkpoint 2 adds ConditionalPrior for iVAE).
      - adaptive-beta schedule: up when raw KL exceeds kl_target, hold
        when recon stops improving, down when KL collapses below kl_floor.
      - dual checkpoint selection:
          * crl_best.pth            → best val_ref_elbo (= val_recon + val_raw_kl)
          * crl_best_aux_type.pth   → best val_aux_type_f1 (epoch-invariant)

    The Prior gets y=(presence, type) labels on every call. StandardPrior
    ignores them; ConditionalPrior in Checkpoint 2 uses them. Plumbing
    y through now makes Checkpoint 2 purely additive.
    """

    # Checkpoint file names.
    CKPT_REF_ELBO     = "crl_best.pth"
    CKPT_AUX_TYPE_F1  = "crl_best_aux_type.pth"

    def __init__(self, prior: Prior, config) -> None:
        super().__init__()
        self.prior  = prior
        self.config = config

    # ------------------------------------------------------------------
    # Prior-aware KL — uses injected prior; beta applied at call site.
    # ------------------------------------------------------------------

    def _kl_terms(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        det_t: torch.Tensor,
        type_t: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (beta*KL, raw_KL) using the injected Prior.

        y = stack(det_t.float(), type_t.float()) — (B, 2). StandardPrior
        ignores it; ConditionalPrior (Checkpoint 2) consumes it.
        """
        y = torch.stack([det_t.float(), type_t.float()], dim=-1)
        raw_kl = self.prior.kl_to_posterior(mu, logvar, y=y)
        return beta * raw_kl, raw_kl

    # ------------------------------------------------------------------
    # Forward pair — dispatches by frontend_type.
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

        det_t  = batch["detection_label_t"][avail].float().to(dev)
        type_t = batch["vehicle_type_t"][avail].to(dev)
        kl, raw_kl = self._kl_terms(mu_t, lv_t, det_t, type_t, beta)

        z_pres, z_type, z_prox, z_env, _ = model.latent.split(z_t)

        aux_pres_logit = model.aux_pres_heads["fused"](z_pres).squeeze(-1)
        aux_pres = F.binary_cross_entropy_with_logits(aux_pres_logit, det_t)

        valid_type = type_t >= 0
        aux_type = torch.tensor(0.0, device=dev)
        aux_type_logit_valid = torch.empty(0, device=dev)
        type_labels_valid    = torch.empty(0, dtype=torch.long, device=dev)
        if valid_type.any():
            aux_type_logit_valid = model.aux_type_heads["fused"](z_type[valid_type])
            type_labels_valid    = type_t[valid_type].long()
            aux_type = F.cross_entropy(aux_type_logit_valid, type_labels_valid)

        n_partners = batch["n_partners"]
        interv = torch.tensor(0.0, device=dev)
        if n_partners > 0:
            x_a_p0 = batch["x_audio_p0"][avail].to(dev)
            x_s_p0 = batch["x_seismic_p0"][avail].to(dev)
            _, z_tn, _, _ = model.encode_fused(x_a_p0, x_s_p0)
            _, _, _, z_env_tn, _ = model.latent.split(z_tn)
            det_tn  = batch["detection_label_p0"][avail].to(dev)
            type_tn = batch["vehicle_type_p0"][avail].to(dev)
            targets = label_change_target(det_t.long(), det_tn, type_t, type_tn).to(dev)
            logits  = model.interv_classifier(z_env, z_env_tn)
            interv  = intervention_matching_loss(logits, targets)

        total = (recon + kl
                 + cfg.lambda_interv   * interv
                 + cfg.lambda_aux_pres * aux_pres
                 + cfg.lambda_aux_type * aux_type)

        metrics = {
            "recon":  recon.item(),
            "kl":     kl.item(),
            "raw_kl": raw_kl.item(),
            "interv": interv.item() if isinstance(interv, torch.Tensor) else interv,
            "total":  total.item(),
            "aux_pres_logits": aux_pres_logit.detach().cpu(),
            "aux_pres_labels": det_t.detach().long().cpu(),
            "aux_type_logits": aux_type_logit_valid.detach().cpu(),
            "aux_type_labels": type_labels_valid.detach().cpu(),
        }
        return total, metrics

    def _forward_pair_per_sensor(
        self, model, batch, beta, dev
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.config

        total_loss = torch.tensor(0.0, device=dev)
        agg: dict[str, float] = {"recon": 0.0, "kl": 0.0, "raw_kl": 0.0,
                                  "interv": 0.0, "total": 0.0}
        aux_pres_logits_all: list[torch.Tensor] = []
        aux_pres_labels_all: list[torch.Tensor] = []
        aux_type_logits_all: list[torch.Tensor] = []
        aux_type_labels_all: list[torch.Tensor] = []
        n_active = 0

        for sensor in model.sensors:
            avail = batch[f"{sensor}_avail"].bool()
            if not avail.any():
                continue

            x = batch[f"x_{sensor}_t"][avail].to(dev)
            features, z_t, mu_t, lv_t = model.encode(sensor, x)
            x_hat = model.decode(sensor, z_t)

            recon  = reconstruction_loss(x_hat, features.detach())

            det_t  = batch["detection_label_t"][avail].float().to(dev)
            type_t = batch["vehicle_type_t"][avail].to(dev)
            kl, raw_kl = self._kl_terms(mu_t, lv_t, det_t, type_t, beta)

            z_pres, z_type, z_prox, z_env, _ = model.latent.split(z_t)

            aux_pres_logit = model.aux_pres_heads[sensor](z_pres).squeeze(-1)
            aux_pres = F.binary_cross_entropy_with_logits(aux_pres_logit, det_t)
            aux_pres_logits_all.append(aux_pres_logit.detach().cpu())
            aux_pres_labels_all.append(det_t.detach().long().cpu())

            valid_type = type_t >= 0
            aux_type = torch.tensor(0.0, device=dev)
            if valid_type.any():
                type_logit_valid  = model.aux_type_heads[sensor](z_type[valid_type])
                type_labels_valid = type_t[valid_type].long()
                aux_type = F.cross_entropy(type_logit_valid, type_labels_valid)
                aux_type_logits_all.append(type_logit_valid.detach().cpu())
                aux_type_labels_all.append(type_labels_valid.detach().cpu())

            interv = torch.tensor(0.0, device=dev)
            n_partners = batch["n_partners"]
            if n_partners > 0:
                x_p0 = batch[f"x_{sensor}_p0"][avail].to(dev)
                _, z_tn, _, _ = model.encode(sensor, x_p0)
                _, _, _, z_env_tn, _ = model.latent.split(z_tn)
                det_tn  = batch["detection_label_p0"][avail].to(dev)
                type_tn = batch["vehicle_type_p0"][avail].to(dev)
                targets = label_change_target(det_t.long(), det_tn, type_t, type_tn).to(dev)
                logits  = model.interv_classifier(z_env, z_env_tn)
                interv  = intervention_matching_loss(logits, targets)

            sensor_loss = (recon + kl
                           + cfg.lambda_interv   * interv
                           + cfg.lambda_aux_pres * aux_pres
                           + cfg.lambda_aux_type * aux_type)
            total_loss = total_loss + sensor_loss
            agg["recon"]  += recon.item()
            agg["kl"]     += kl.item()
            agg["raw_kl"] += raw_kl.item()
            agg["interv"] += interv.item() if isinstance(interv, torch.Tensor) else interv
            agg["total"]  += sensor_loss.item()
            n_active += 1

        if n_active > 1:
            total_loss = total_loss / n_active
            for k in agg:
                agg[k] /= n_active
        if n_active == 0:
            total_loss = torch.tensor(0.0, device=dev, requires_grad=True)

        agg["aux_pres_logits"] = (
            torch.cat(aux_pres_logits_all) if aux_pres_logits_all else torch.empty(0)
        )
        agg["aux_pres_labels"] = (
            torch.cat(aux_pres_labels_all) if aux_pres_labels_all
            else torch.empty(0, dtype=torch.long)
        )
        agg["aux_type_logits"] = (
            torch.cat(aux_type_logits_all) if aux_type_logits_all else torch.empty(0)
        )
        agg["aux_type_labels"] = (
            torch.cat(aux_type_labels_all) if aux_type_labels_all
            else torch.empty(0, dtype=torch.long)
        )
        return total_loss, agg

    # ------------------------------------------------------------------
    # Derived val metrics, beta, checkpoint selection.
    # ------------------------------------------------------------------

    def val_metrics_summary(self, val_m: dict) -> dict:
        """val_ref_elbo = val_recon + val_raw_kl — epoch-invariant at beta=1."""
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
        """Backward-compatible summary matching the existing crl_checkpoint_summary.json."""
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
