"""
LR scheduler: linear warmup followed by cosine decay.

build_scheduler(optimizer, config) returns a LambdaLR that:
  - Linearly ramps LR from 0 → config.lr over config.warmup_epochs.
  - Then follows cosine decay from config.lr → config.lr_min over the
    remaining epochs (T_max = n_epochs - warmup_epochs).

Call scheduler.step() once per epoch (not per batch).
No restarts — one clean annealing cycle for simplicity.
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from crl_vehicle.config import CRLConfig


def build_scheduler(optimizer: Optimizer, config: CRLConfig) -> LambdaLR:
    warmup = config.warmup_epochs
    T_max = max(config.n_epochs - warmup, 1)
    lr_min = config.lr_min
    lr_max = config.lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(warmup, 1))
        t = (epoch - warmup) / T_max
        t = min(t, 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        scaled = (lr_min + cos_factor * (lr_max - lr_min)) / lr_max
        return float(scaled)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
