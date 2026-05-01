"""Deterministic seeding for reproducible runs.

Seeds Python `random`, numpy, and torch (CPU + CUDA) global RNGs, and exposes
a helper for seeding DataLoader workers + shuffle order so reruns of the same
config produce bit-identical training trajectories.

Two seeds are derived from the user-supplied seed:
  - the global seed (used directly), which fixes model init, augmentation,
    reparameterization noise, and partner sampling.
  - a DataLoader generator seeded with the same value, which fixes shuffle
    order across epochs.
  - per-worker seeds derived as `seed + worker_id` so each worker has its
    own deterministic stream that doesn't collide with other workers'.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python `random`, numpy, and torch (CPU + CUDA if available).

    Call once at process start, after argparse, before any model/data
    construction. Idempotent.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn_factory(seed: int):
    """Return a worker_init_fn that seeds each DataLoader worker
    deterministically as `seed + worker_id`."""

    def _init(worker_id: int) -> None:
        wseed = seed + worker_id
        random.seed(wseed)
        np.random.seed(wseed)
        torch.manual_seed(wseed)

    return _init


def seeded_dataloader_kwargs(seed: int) -> dict:
    """DataLoader kwargs that fix shuffle order + worker RNG.

    Spread into `DataLoader(...)` alongside the existing args:
        DataLoader(ds, ..., **seeded_dataloader_kwargs(args.seed))
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "generator": g,
        "worker_init_fn": _worker_init_fn_factory(seed),
    }
