#!/usr/bin/env python3
"""
plot_aggregate.py — cross-run comparison plots (poster-styled, top-5 only).

Generates four figures that overlay data from the top-5 runs (ranked by
best_type_f1, so the same five runs appear in every figure for visual
continuity across the poster):

    val_type_f1_over_epochs — top-5 val_aux_type_f1 trajectories
    val_ref_elbo_over_epochs — top-5 val_ref_elbo trajectories (y-clipped)
    best_f1_by_run          — top-5 horizontal bar chart, best type_f1
    complexity_vs_f1        — top-5 scatter, epochs-to-best vs best type_f1

Each run gets a distinct color (POSTER_PALETTE) + a linestyle that encodes
its frontend family (assigned dynamically per-figure).

Usage
-----
    python plot_aggregate.py
    python plot_aggregate.py --filter frontend_type=multiscale
    python plot_aggregate.py --include-diverged
    python plot_aggregate.py --top-n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A
from crl_vehicle import plotting as P


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_metric_over_epochs(
    runs: list[A.RunMetrics],
    styles: list[dict],
    metric: str,
    title: str,
    out: Path,
    clip_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for rm, s in zip(runs, styles, strict=True):
        ts = A.load_crl_timeseries(rm.path)
        values = ts.get(metric, [])
        if not values:
            continue
        epochs = ts.get("epoch", list(range(len(values))))
        ax.plot(
            epochs[: len(values)],
            values,
            color=s["color"],
            linestyle=s["linestyle"],
            label=s["label"],
        )
        plotted += 1
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if clip_max is not None:
        ax.set_ylim(0, clip_max)
    if plotted > 0:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    P.poster_save(fig, out)


def _label_with_probe(label: str, probe: str | None) -> str:
    """Append the type-head winning probe to a bar label, abbreviated.
    'mlp_ztype__crl_best_aux_type' -> '(mlp_ztype_aux)'."""
    if not probe:
        return label
    name, _, ckpt = probe.partition("__")
    suffix = "_aux" if ckpt.endswith("aux_type") else ""
    return f"{label}\n({name}{suffix})"


def plot_best_f1_bar(runs: list[A.RunMetrics], styles: list[dict], out: Path) -> None:
    pairs = [(rm, s) for rm, s in zip(runs, styles, strict=True) if rm.best_type_f1 is not None]
    if not pairs:
        print("  (bar chart skipped — no runs have best_type_f1)")
        return
    # Plot bars in ascending order so best ends up at the top.
    pairs = sorted(pairs, key=lambda p: p[0].best_type_f1)
    names = [_label_with_probe(s["label"], rm.best_type_probe) for rm, s in pairs]
    values = [rm.best_type_f1 for rm, _ in pairs]
    colors = [s["color"] for _, s in pairs]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.9 * len(names))))
    ax.barh(names, values, color=colors, alpha=0.85)
    ax.axvline(
        A.SHIP_TYPE_F1,
        color="red",
        linestyle="--",
        label=f"ship threshold ({A.SHIP_TYPE_F1})",
    )
    ax.set_xlabel("best val_type_f1 (downstream probe)")
    ax.set_xlim(0, max(1.0, max(values) + 0.05))
    ax.set_title("Best downstream type_f1 — top 5 runs")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    P.poster_save(fig, out)


def plot_complexity_vs_f1(runs: list[A.RunMetrics], styles: list[dict], out: Path) -> None:
    """Scatter: x = epochs to best, y = best type_f1. Lower-right quadrant
    is the 'cheap + good' sweet spot."""
    pairs = [
        (rm, s)
        for rm, s in zip(runs, styles, strict=True)
        if rm.best_type_f1 is not None and rm.best_aux_type_epoch is not None
    ]
    if not pairs:
        print("  (complexity scatter skipped — no runs with both metrics)")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    for rm, s in pairs:
        ax.scatter(
            rm.epochs_completed,
            rm.best_type_f1,
            color=s["color"],
            s=200,
            edgecolor="black",
            linewidth=1.5,
            label=s["label"],
        )
        ax.annotate(
            rm.name,
            (rm.epochs_completed, rm.best_type_f1),
            xytext=(6, 6),
            textcoords="offset points",
            fontweight="bold",
        )
    ax.axhline(A.SHIP_TYPE_F1, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("epochs completed (CRL)")
    ax.set_ylabel("best downstream type_f1")
    ax.set_title("Cost vs quality — top 5 runs")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    P.poster_save(fig, out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="saved_crl/runs", type=Path)
    p.add_argument("--out", default="saved_crl/analysis", type=Path)
    p.add_argument("--filter", action="append", default=[], metavar="key=val")
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--top-n", type=int, default=5, help="Number of top runs to plot")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    P.apply_poster_style()

    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.root)]
    if not runs:
        print(f"No runs found under {args.root}", file=sys.stderr)
        return 1

    filters = A.parse_filter_args(args.filter)
    runs = A.apply_filters(runs, filters, exclude_diverged=not args.include_diverged)

    # Single ranking — best_type_f1 — drives the top-5 selection for every
    # plot in this script. Same 5 runs appear in every figure so a poster
    # viewer can track a run by color across plots.
    top = P.top_n_runs(runs, n=args.top_n, by="best_type_f1")
    if not top:
        print("No runs with best_type_f1; nothing to plot.", file=sys.stderr)
        return 1

    styles = P.assign_run_styles(top)
    print(f"Plotting top-{len(top)} runs:")
    for rm in top:
        probe = rm.best_type_probe or "?"
        print(
            f"  {rm.name}  best_type_f1={rm.best_type_f1:.3f}  "
            f"({rm.config.get('frontend_type','?')}, via {probe})"
        )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_over_epochs(
        top,
        styles,
        "val_aux_type_f1",
        title="val_aux_type_f1 — top 5 runs",
        out=out_dir / "val_type_f1_over_epochs",
    )
    # Diverged runs would have ELBO > 1000; top_n_runs already excludes them.
    plot_metric_over_epochs(
        top,
        styles,
        "val_ref_elbo",
        title="val_ref_elbo — top 5 runs (y-clip at 20)",
        out=out_dir / "val_ref_elbo_over_epochs",
        clip_max=20.0,
    )
    plot_best_f1_bar(top, styles, out_dir / "best_f1_by_run")
    plot_complexity_vs_f1(top, styles, out_dir / "complexity_vs_f1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
