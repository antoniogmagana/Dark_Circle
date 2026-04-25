"""Manifest builder for the ID split schema (--use-id-split).

Computes per-group window-range manifests from DATASET_VEHICLE_MAP
markers ("split", "split_runs"). Designed to be pure (no torch, no
CUDA) and cache-friendly.

See docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
"""
from __future__ import annotations


def compute_split_intervals(n_paired: int) -> dict[str, list[tuple[int, int]]] | None:
    """Half/half split on paired window count for "split" marker.

    Returns:
        {"val": [(0, n_paired // 2)], "test": [(n_paired // 2, n_paired)]}
        or None if n_paired < 2 (cannot split).

    Intervals are half-open [start, end) in paired-window coordinates.
    """
    if n_paired < 2:
        return None
    half = n_paired // 2
    return {
        "val":  [(0, half)],
        "test": [(half, n_paired)],
    }
