#!/usr/bin/env python3
"""
plot_aggregate.py — cross-run comparison plots.

Generates four figures that overlay data from multiple runs:

    val_type_f1_over_epochs — every run's val_aux_type_f1 trajectory
    val_ref_elbo_over_epochs — every run's val_ref_elbo trajectory (y-clipped)
    best_f1_by_run          — horizontal bar chart, best type_f1 per run
    complexity_vs_f1        — scatter of epochs-to-best vs best type_f1

Color is consistent per frontend across figures (see analysis.FRONTEND_COLORS).

Usage
-----
    python plot_aggregate.py
    python plot_aggregate.py --filter frontend_type=multiscale
    python plot_aggregate.py --include-diverged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------

def _save(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=120)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out_stem.with_suffix('.png')}")


def _legend_inside_or_outside(ax, n_entries: int) -> None:
    """Small legend inline; large legend to the side."""
    if n_entries <= 6:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left",
                  borderaxespad=0.0)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_metric_over_epochs(
    runs: list[A.RunMetrics], metric: str, title: str,
    out: Path, clip_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for rm in runs:
        ts = A.load_crl_timeseries(rm.path)
        values = ts.get(metric, [])
        if not values:
            continue
        epochs = ts.get("epoch", list(range(len(values))))
        color = A.frontend_color(rm)
        ax.plot(epochs[:len(values)], values, color=color, alpha=0.7,
                linewidth=1.2, label=rm.name)
        plotted += 1
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if clip_max is not None:
        ax.set_ylim(0,clip_max)
    if plotted > 0:
        _legend_inside_or_outside(ax, plotted)
    fig.tight_layout()
    _save(fig, out)


def plot_best_f1_bar(runs: list[A.RunMetrics], out: Path) -> None:
    runs_with_f1 = [rm for rm in runs if rm.best_type_f1 is not None]
    if not runs_with_f1:
        print("  (bar chart skipped — no runs have best_type_f1)")
        return
    runs_with_f1.sort(key=lambda r: r.best_type_f1)
    names = [rm.name for rm in runs_with_f1]
    values = [rm.best_type_f1 for rm in runs_with_f1]
    colors = [A.frontend_color(rm) for rm in runs_with_f1]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(names))))
    ax.barh(names, values, color=colors, alpha=0.8)
    ax.axvline(A.SHIP_TYPE_F1, color="red", linestyle="--", linewidth=1,
               label=f"ship threshold ({A.SHIP_TYPE_F1})")
    ax.set_xlabel("best val_type_f1 (downstream probe)")
    ax.set_xlim(0, max(1.0, max(values) + 0.05))
    ax.set_title("Best downstream type_f1 by run")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)

    # Frontend legend — one swatch per unique frontend.
    seen = {}
    for rm in runs_with_f1:
        fe = rm.config.get("frontend_type", "")
        if fe not in seen:
            seen[fe] = A.frontend_color(rm)
    patches = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8, label=fe)
        for fe, c in seen.items()
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"ship threshold ({A.SHIP_TYPE_F1})"),
    ], loc="lower right", fontsize=8)

    fig.tight_layout()
    _save(fig, out)


def plot_complexity_vs_f1(runs: list[A.RunMetrics], out: Path) -> None:
    """Scatter: x = epochs to best, y = best type_f1. Lower-right quadrant
    is the 'cheap + good' sweet spot."""
    points = []
    for rm in runs:
        if rm.best_type_f1 is None or rm.best_aux_type_epoch is None:
            continue
        # best_aux_type_epoch tracks best aux F1; use epochs_completed as
        # a fallback proxy for downstream F1 convergence (they're correlated
        # but not identical). The downstream best epoch isn't in summary.
        points.append((rm.epochs_completed, rm.best_type_f1, rm))
    if not points:
        print("  (complexity scatter skipped — no runs with both metrics)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_labels = set()
    for ep, f1, rm in points:
        color = A.frontend_color(rm)
        fe = rm.config.get("frontend_type", "")
        label = fe if fe not in seen_labels else None
        seen_labels.add(fe)
        ax.scatter(ep, f1, color=color, s=80, alpha=0.8, edgecolor="black",
                   linewidth=0.5, label=label)
        ax.annotate(rm.name, (ep, f1), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(A.SHIP_TYPE_F1, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("epochs completed (CRL)")
    ax.set_ylabel("best downstream type_f1")
    ax.set_title("Cost vs quality: epochs to train vs best type_f1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root",   default="saved_crl/runs", type=Path)
    p.add_argument("--out",    default="saved_crl/analysis", type=Path)
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

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_over_epochs(
        runs, "val_aux_type_f1",
        title="val_aux_type_f1 across runs",
        out=out_dir / "val_type_f1_over_epochs",
    )
    # Diverged runs can have ELBO > 1000; clip y-axis for visibility.
    plot_metric_over_epochs(
        runs, "val_ref_elbo",
        title="val_ref_elbo across runs (y-clip at 10)",
        out=out_dir / "val_ref_elbo_over_epochs",
        clip_max=20.0,
    )
    plot_best_f1_bar(runs, out_dir / "best_f1_by_run")
    plot_complexity_vs_f1(runs, out_dir / "complexity_vs_f1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
