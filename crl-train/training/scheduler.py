"""
LR scheduler: linear warmup followed by CosineAnnealingWarmRestarts.

build_scheduler(optimizer, config) returns a single LambdaLR that wraps
both phases so the trainer only calls scheduler.step() once per epoch.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

from crl_vehicle.config import CRLConfig


def build_scheduler(
    optimizer: Optimizer, config: CRLConfig
) -> LambdaLR:
    """
    Returns a LambdaLR that:
      - Linearly ramps LR from 0 → config.lr over config.warmup_epochs.
      - Then follows CosineAnnealingWarmRestarts with T_0=config.cosine_period
        and eta_min=config.lr_min.

    Call scheduler.step() once per epoch (not per batch).
    """
    warmup = config.warmup_epochs
    T0 = config.cosine_period
    lr_min = config.lr_min
    lr_max = config.lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            # Linear ramp: fraction of full LR
            return float(epoch + 1) / float(max(warmup, 1))

        # Cosine annealing (restart-aware)
        # Mimics CosineAnnealingWarmRestarts without needing a separate scheduler.
        epoch_in_cycle = (epoch - warmup) % T0
        cos_factor = 0.5 * (1.0 + torch.cos(
            torch.tensor(3.14159265358979 * epoch_in_cycle / T0)
        ).item())
        # Scale relative to lr_max (LambdaLR multiplies the base LR)
        scaled = (lr_min + cos_factor * (lr_max - lr_min)) / lr_max
        return float(scaled)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
