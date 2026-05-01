#!/usr/bin/env python3
"""
compare_cross_location.py — per-dataset F1 comparison + ship-metric ranking.

The capstone deliverable requires cross-location generalization. A model that
scores 0.75 val_type_f1 by dominating focal and failing m3nvc is NOT
shippable; one that scores 0.70 by being mediocre everywhere IS. So the ship
metric is min-across-datasets type_f1, not the aggregate.

Usage
-----
    python compare_cross_location.py
    python compare_cross_location.py --filter frontend_type=multiscale

Outputs (default --out saved_crl/analysis/):
    cross_location.csv         — run × dataset matrix
    cross_location.md          — markdown table sorted by min-F1
    cross_location_heatmap.png — matplotlib heatmap
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, int | float) else ""


# --------------------------------------------------------------------------
# CSV / Markdown
# --------------------------------------------------------------------------


def collect_matrix(
    runs: list[A.RunMetrics],
) -> tuple[list[str], list[str], np.ndarray]:
    """Return (run_names, dataset_names, matrix) with matrix[i,j] = type_f1
    for run i on dataset j. Cells for (run, dataset) with no eval report
    become NaN."""
    all_datasets: set[str] = set()
    for rm in runs:
        all_datasets.update(rm.per_dataset_type_f1.keys())
    dataset_names = sorted(all_datasets)
    run_names = [rm.name for rm in runs]
    mat = np.full((len(runs), len(dataset_names)), np.nan)
    for i, rm in enumerate(runs):
        for j, ds in enumerate(dataset_names):
            v = rm.per_dataset_type_f1.get(ds)
            if v is not None:
                mat[i, j] = v
    return run_names, dataset_names, mat


def write_csv(path: Path, runs: list[A.RunMetrics], ds_names: list[str], mat: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frontend_type", *ds_names, "min_type_f1", "worst_dataset"])
        for i, rm in enumerate(runs):
            row = [rm.name, rm.config.get("frontend_type", "")]
            row += [
                f"{mat[i, j]:.4f}" if not np.isnan(mat[i, j]) else "" for j in range(len(ds_names))
            ]
            row.append(
                f"{rm.min_dataset_type_f1:.4f}" if rm.min_dataset_type_f1 is not None else ""
            )
            row.append(rm.worst_dataset or "")
            w.writerow(row)


def render_markdown(
    runs: list[A.RunMetrics],
    ds_names: list[str],
    mat: np.ndarray,
) -> str:
    lines = [
        "# Cross-Location Type F1",
        "",
        "Ship metric is `min_type_f1` (worst case across datasets).",
        "",
    ]
    header = ["Run", "Frontend", *ds_names, "min_F1", "worst"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    # Sort runs by min_type_f1 desc.
    order = sorted(
        range(len(runs)),
        key=lambda i: (
            -(runs[i].min_dataset_type_f1 if runs[i].min_dataset_type_f1 is not None else -1)
        ),
    )
    for i in order:
        rm = runs[i]
        row = [rm.name, rm.config.get("frontend_type", "")]
        row += [_fmt(mat[i, j]) if not np.isnan(mat[i, j]) else "—" for j in range(len(ds_names))]
        row.append(_fmt(rm.min_dataset_type_f1))
        row.append(rm.worst_dataset or "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Heatmap
# --------------------------------------------------------------------------


def render_heatmap(
    runs: list[A.RunMetrics],
    ds_names: list[str],
    mat: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs or not ds_names:
        print("  (heatmap skipped — no runs or no datasets with eval data)")
        return

    # Sort runs by min_type_f1 desc so best row is at top.
    order = sorted(
        range(len(runs)),
        key=lambda i: -(
            runs[i].min_dataset_type_f1 if runs[i].min_dataset_type_f1 is not None else -1
        ),
    )
    sorted_mat = mat[order]
    sorted_names = [runs[i].name for i in order]

    fig, ax = plt.subplots(
        figsize=(max(4, 1 + 0.8 * len(ds_names)), max(3, 0.5 * len(sorted_names))),
    )
    im = ax.imshow(sorted_mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_title("Cross-location type_f1\n(sorted by min-across-datasets, best at top)")

    # Annotate cells with their values.
    for i in range(sorted_mat.shape[0]):
        for j in range(sorted_mat.shape[1]):
            v = sorted_mat[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="type_f1")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=120)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="saved_crl/runs", type=Path)
    p.add_argument("--out", default="saved_crl/analysis", type=Path)
    p.add_argument("--filter", action="append", default=[], metavar="key=val")
    p.add_argument("--include-diverged", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.root)]
    if not runs:
        print(f"No runs found under {args.root}", file=sys.stderr)
        return 1

    filters = A.parse_filter_args(args.filter)
    runs = A.apply_filters(runs, filters, exclude_diverged=not args.include_diverged)
    # Drop runs with no per-dataset eval at all — they contribute no information.
    runs_with_eval = [rm for rm in runs if rm.per_dataset_type_f1]
    if not runs_with_eval:
        print(
            "No runs have per-dataset eval data. " "Did you run run_full_diagnostic.py?",
            file=sys.stderr,
        )
        return 1

    run_names, ds_names, mat = collect_matrix(runs_with_eval)
    args.out.mkdir(parents=True, exist_ok=True)

    write_csv(args.out / "cross_location.csv", runs_with_eval, ds_names, mat)
    (args.out / "cross_location.md").write_text(render_markdown(runs_with_eval, ds_names, mat))
    render_heatmap(runs_with_eval, ds_names, mat, args.out / "cross_location_heatmap")

    print(f"Wrote {args.out / 'cross_location.csv'}")
    print(f"Wrote {args.out / 'cross_location.md'}")
    print(f"Wrote {args.out / 'cross_location_heatmap.png'} (and .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
