#!/usr/bin/env python3
"""
compare_ablations.py — pairwise deltas along predefined ablation axes.

For each axis in ABLATION_AXES (frontend_type, morlet_use_phase, prior_type,
training_mode, stage2, morlet_learnable_w0), this script finds every pair of
runs that differ ONLY in that axis (all other axes match) and reports
Δpres_f1, Δtype_f1, Δref_elbo.

best_pres_f1 and best_type_f1 each come from the per-head winning probe
within their run (see crl_vehicle/analysis.py:_best_probe_per_head). The
two sides of a pair may therefore have different winning probes; that's
expected — we're comparing each run's best, not a fixed probe across runs.

This is how you answer "does knob X matter?" without manually eyeballing the
leaderboard.

Usage
-----
    python compare_ablations.py
    python compare_ablations.py --axes morlet_use_phase stage2
    python compare_ablations.py --out saved_crl/analysis/ablations_focus.md

Output:
    ablations.md — one section per axis with a delta table, or "no pairs".
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A

# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def find_pairs_along_axis(
    runs: list[A.RunMetrics],
    axis: str,
    all_axes: tuple[str, ...],
) -> list[tuple[A.RunMetrics, A.RunMetrics]]:
    """Return list of (baseline, variant) pairs where `axis` differs and all
    other axes match. Ordering within a pair is arbitrary; the report
    normalizes presentation later."""
    pairs: list[tuple[A.RunMetrics, A.RunMetrics]] = []
    other_axes = tuple(a for a in all_axes if a != axis)
    for a, b in combinations(runs, 2):
        sig_a = A.axis_signature(a)
        sig_b = A.axis_signature(b)
        if sig_a[axis] == sig_b[axis]:
            continue
        # Skip pairs where either side has a missing axis value — those are
        # runs that predate the field, not real ablation pairs.
        if sig_a[axis] is None or sig_b[axis] is None:
            continue
        # All non-axis signatures must match.
        if any(sig_a[x] != sig_b[x] for x in other_axes):
            continue
        pairs.append((a, b))
    return pairs


def _delta(x: float | None, y: float | None) -> str:
    if x is None or y is None:
        return "—"
    return f"{y - x:+.4f}"


def _fmt_run(rm: A.RunMetrics) -> str:
    return rm.name


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_axis_section(axis: str, pairs: list[tuple[A.RunMetrics, A.RunMetrics]]) -> str:
    lines = [f"## {axis}", ""]
    if not pairs:
        lines.append("_No matching pairs found._")
        lines.append("")
        return "\n".join(lines)
    lines.append(
        "| Baseline | Variant | Axis: base → variant | " "Δpres_f1 | Δtype_f1 | Δref_elbo |"
    )
    lines.append("|---|---|---|---|---|---|")
    for a, b in pairs:
        # Pick canonical ordering: baseline = lexicographically lower axis value.
        va = A._lookup_run_value(a, axis)
        vb = A._lookup_run_value(b, axis)
        if str(va) > str(vb):
            a, b = b, a
            va, vb = vb, va
        lines.append(
            f"| {_fmt_run(a)} | {_fmt_run(b)} | "
            f"{va} → {vb} | "
            f"{_delta(a.best_pres_f1, b.best_pres_f1)} | "
            f"{_delta(a.best_type_f1, b.best_type_f1)} | "
            f"{_delta(a.best_val_ref_elbo, b.best_val_ref_elbo)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(
    axes: tuple[str, ...],
    runs: list[A.RunMetrics],
) -> str:
    header = [
        "# Ablation Pairs",
        "",
        f"Runs considered: {len(runs)}",
        "",
        "A pair matches an axis when those two runs differ in that axis "
        "AND all other tracked axes are identical. Δ = variant − baseline.",
        "",
    ]
    sections = []
    for axis in axes:
        pairs = find_pairs_along_axis(runs, axis, A.ABLATION_AXES)
        sections.append(render_axis_section(axis, pairs))
    return "\n".join(header + sections)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="saved_crl/runs", type=Path)
    p.add_argument("--out", default="saved_crl/analysis/ablations.md", type=Path)
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="key=val",
        help="Filter runs by config field before pairing (repeatable).",
    )
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument(
        "--axes",
        nargs="+",
        default=list(A.ABLATION_AXES),
        help=f"Axes to pair along. Default: all of {A.ABLATION_AXES}.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.root)]
    if not runs:
        print(f"No runs found under {args.root}", file=sys.stderr)
        return 1

    filters = A.parse_filter_args(args.filter)
    runs = A.apply_filters(runs, filters, exclude_diverged=not args.include_diverged)

    unknown = [a for a in args.axes if a not in A.ABLATION_AXES]
    if unknown:
        print(
            f"Unknown axes: {unknown}. Valid: {list(A.ABLATION_AXES)}",
            file=sys.stderr,
        )
        return 1

    report = render_report(tuple(args.axes), runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
