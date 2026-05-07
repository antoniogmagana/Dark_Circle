"""Standalone replot of val_ref_elbo_over_epochs with a tighter y-clip.

The aggregator's `plot_aggregate.py` clips this figure at y=20, which makes the
post-spike convergence behavior unreadable. This script regenerates only that
one figure at clip=10 (without touching the bar chart, scatter, or
val_type_f1 figure) and writes alongside the original.

Usage:
    poetry run python replot_ref_elbo.py [--clip 10]

Reads exactly the same data and runs the aggregator selects (top-5 by best
downstream type_f1, non-diverged), so the lines and colors match figure 2 in
the same Doc 2 section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# Use the same module the aggregator uses so we don't drift from its
# top-5 selection or its color/style assignment.
import sys

CRL_TRAIN = Path(__file__).resolve().parents[2] / "crl-train"
sys.path.insert(0, str(CRL_TRAIN))

from crl_vehicle import analysis as A  # noqa: E402
from crl_vehicle import plotting as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=CRL_TRAIN / "saved_crl" / "analysis" / "val_ref_elbo_over_epochs_clip10",
        help="Output stem (PNG + PDF written by P.poster_save).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=CRL_TRAIN / "saved_crl" / "runs",
    )
    args = ap.parse_args()

    P.apply_poster_style()
    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.runs_root)]
    runs = A.apply_filters(runs, {}, exclude_diverged=True)
    top = P.top_n_runs(runs, n=5, by="best_type_f1")
    if not top:
        print("no runs to plot")
        return 1
    styles = P.assign_run_styles(top)
    print(f"Plotting top-{len(top)} runs (clip={args.clip}):")
    for rm in top:
        print(f"  {rm.name}  best_type_f1={rm.best_type_f1:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for rm, s in zip(top, styles, strict=True):
        ts = A.load_crl_timeseries(rm.path)
        values = ts.get("val_ref_elbo")
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
    ax.set_ylabel("val_ref_elbo")
    ax.set_title(f"val_ref_elbo — top 5 runs (y-clip at {args.clip:g})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, args.clip)
    if plotted:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    P.poster_save(fig, args.out)
    print(f"wrote {args.out}.png and {args.out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
