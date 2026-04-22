"""Inference-time log-prior shift for classifier heads.

When a probe is trained with class-balanced loss (pos_weight or class_weights),
the effective training prior is approximately uniform. Evaluating on a split
with a different class distribution biases predictions toward whichever class
dominates the split. The log-prior shift corrects this:

    logit_adjusted = logit - log(p_train) + log(p_split)

This yields the classification the representation would give under the
split's true prior. Use only for diagnostic reporting with labels known
(oracle prior); not appropriate for deployment, where target priors are
unknown. Report column names must carry the `_target_calibrated` suffix
to prevent confusion with uncalibrated deployment numbers.
"""
from __future__ import annotations

import math

import torch


UNIFORM_BINARY_PRIOR: float = 0.5


def uniform_multiclass_prior(n_classes: int) -> torch.Tensor:
    return torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float32)


def compute_binary_prior(
    labels: torch.Tensor, eps: float = 1e-8
) -> float:
    """Empirical P(y=1) from a tensor of {0,1} labels."""
    if labels.numel() == 0:
        return UNIFORM_BINARY_PRIOR
    p = labels.float().mean().item()
    return min(max(p, eps), 1.0 - eps)


def compute_multiclass_prior(
    labels: torch.Tensor, n_classes: int, eps: float = 1e-8
) -> torch.Tensor:
    """Empirical class distribution from a tensor of integer labels."""
    if labels.numel() == 0:
        return uniform_multiclass_prior(n_classes)
    counts = torch.bincount(labels.long(), minlength=n_classes).float()
    p = counts / counts.sum().clamp(min=1.0)
    return p.clamp(min=eps)


def apply_binary_log_prior_shift(
    logits: torch.Tensor,
    p_split: float,
    p_train: float = UNIFORM_BINARY_PRIOR,
) -> torch.Tensor:
    """Shift binary logits so threshold=0 corresponds to the target-split prior.

    For BCE logits (logit = log P(y=1) - log P(y=0)), the shift is
    Δ = log(p_split/(1-p_split)) - log(p_train/(1-p_train)).
    """
    def _lodds(p: float) -> float:
        return math.log(p / (1.0 - p))
    return logits + (_lodds(p_split) - _lodds(p_train))


def apply_multiclass_log_prior_shift(
    logits: torch.Tensor,
    p_split: torch.Tensor,
    p_train: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shift K-class logits so argmax reflects the target-split prior.

    logit_adjusted[c] = logit[c] - log(p_train[c]) + log(p_split[c])

    p_split, p_train are (K,) probability vectors over classes. p_train
    defaults to uniform (which matches class-weighted training).
    """
    n_classes = logits.shape[-1]
    if p_split.shape[-1] != n_classes:
        raise ValueError(
            f"p_split has {p_split.shape[-1]} classes, logits has {n_classes}"
        )
    if p_train is None:
        p_train = uniform_multiclass_prior(n_classes)
    if p_train.shape[-1] != n_classes:
        raise ValueError(
            f"p_train has {p_train.shape[-1]} classes, logits has {n_classes}"
        )
    shift = torch.log(p_split.to(logits.dtype).to(logits.device)) \
          - torch.log(p_train.to(logits.dtype).to(logits.device))
    return logits + shift
