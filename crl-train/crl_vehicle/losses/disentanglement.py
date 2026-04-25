from __future__ import annotations

import torch


def cross_modal_alignment_loss(
    mu_signal_audio: torch.Tensor,
    mu_signal_seismic: torch.Tensor,
) -> torch.Tensor:
    """L2 distance between per-modality signal estimates.

    Encodes the cross-modal coherence prior: a vehicle excites both sensors
    with the same source, so the signal subspace should agree across audio
    and seismic. Discrepancy is what one modality encodes that the other
    doesn't — by definition non-shared, treated as noise.

    mu_signal_audio:   (B, d_signal)
    mu_signal_seismic: (B, d_signal)
    Returns: scalar (mean of squared L2 over batch).
    """
    if mu_signal_audio.shape != mu_signal_seismic.shape:
        raise ValueError(
            f"shape mismatch: audio {mu_signal_audio.shape} vs "
            f"seismic {mu_signal_seismic.shape}"
        )
    return ((mu_signal_audio - mu_signal_seismic) ** 2).sum(dim=-1).mean()


def temporal_stability_loss(
    mu_env_t: torch.Tensor,
    mu_env_tn: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """L2 distance between env latents of consecutive windows.

    Encodes the slow-environment prior: env should vary slowly compared to
    signal. Only consecutive-window partner pairs (STRATUM_CONSEC) contribute.
    Returns 0 if no valid pairs in batch.

    mu_env_t:   (B, d_env)
    mu_env_tn:  (B, d_env)
    valid_mask: (B,) bool
    Returns: scalar (mean over valid rows).
    """
    if not valid_mask.any():
        return torch.zeros((), device=mu_env_t.device, requires_grad=True)
    diff = ((mu_env_t - mu_env_tn) ** 2).sum(dim=-1)
    return diff[valid_mask].mean()


def intervention_invariance_loss(
    mu_signal_clean: torch.Tensor,
    mu_signal_intervened: torch.Tensor,
) -> torch.Tensor:
    """L2 distance between signal latents of clean vs noise-intervened input.

    Encodes the noise-invariance prior: vehicle signal should be unchanged
    when only the environment is perturbed. Pushes intervention-induced
    variation out of z_signal and into z_env.

    mu_signal_clean:      (B, d_signal)
    mu_signal_intervened: (B, d_signal)
    Returns: scalar (mean of squared L2 over batch).
    """
    if mu_signal_clean.shape != mu_signal_intervened.shape:
        raise ValueError(
            f"shape mismatch: clean {mu_signal_clean.shape} vs "
            f"intervened {mu_signal_intervened.shape}"
        )
    return ((mu_signal_clean - mu_signal_intervened) ** 2).sum(dim=-1).mean()
