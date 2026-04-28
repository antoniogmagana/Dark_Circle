#!/usr/bin/env python3
"""
compare_runs.py — leaderboard of completed CRL runs.

Walks saved_crl/runs/ (or --root) recursively, reads each run's meta.json +
crl/crl_metrics.csv + downstream/<probe>/downstream_metrics.csv +
eval/<probe>/<split>/eval_report.json, and emits a sorted leaderboard
as both CSV and markdown.

Usage
-----
    python compare_runs.py
    python compare_runs.py --filter frontend_type=morlet_per_sensor
    python compare_runs.py --include-diverged --sort best_val_ref_elbo
    python compare_runs.py --top 5

Output (default --out saved_crl/analysis/):
    leaderboard.csv   — all columns, one row per run
    leaderboard.md    — markdown table sorted by --sort, shippable rows flagged
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------

COLUMNS = [
    "name",
    "frontend_type",
    "morlet_use_phase",
    "prior_type",
    "training_mode",
    "stage2",
    "epochs_completed",
    "best_pres_f1",
    "best_type_f1",
    "best_val_ref_elbo",
    "min_dataset_type_f1",
    "worst_dataset",
    "calibrated_type_f1",
    "balanced_accuracy",
    "mcc",
    "shippable",
    "diverged",
]


def _row(rm: A.RunMetrics) -> dict:
    return {
        "name":                rm.name,
        "frontend_type":       rm.config.get("frontend_type", ""),
        "morlet_use_phase":    rm.config.get("morlet_use_phase", ""),
        "prior_type":          rm.config.get("prior_type", ""),
        "training_mode":       rm.config.get("training_mode", ""),
        "stage2":              rm.stage2,
        "epochs_completed":    rm.epochs_completed,
        "best_pres_f1":        _fmt(rm.best_pres_f1),
        "best_type_f1":        _fmt(rm.best_type_f1),
        "best_val_ref_elbo":   _fmt(rm.best_val_ref_elbo),
        "min_dataset_type_f1": _fmt(rm.min_dataset_type_f1),
        "worst_dataset":       rm.worst_dataset or "",
        "calibrated_type_f1":  _fmt(rm.calibrated_type_f1),
        "balanced_accuracy":   _fmt(rm.balanced_accuracy),
        "mcc":                 _fmt(rm.mcc),
        "shippable":           rm.shippable,
        "diverged":            rm.diverged,
    }


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 4)
    return v


# --------------------------------------------------------------------------
# Sorting
# --------------------------------------------------------------------------

# Metrics where higher is better vs lower is better.
_SORT_DIRECTION = {
    "best_pres_f1":        "desc",
    "best_type_f1":        "desc",
    "min_dataset_type_f1": "desc",
    "calibrated_type_f1":  "desc",
    "balanced_accuracy":   "desc",
    "mcc":                 "desc",
    "shippable":           "desc",
    "epochs_completed":    "desc",
    "best_val_ref_elbo":   "asc",
}


def _sort_key(row: dict, field: str):
    val = row.get(field)
    if val == "" or val is None:
        # Push missing values to the end regardless of direction.
        return (1, 0)
    if isinstance(val, bool):
        return (0, -int(val) if _SORT_DIRECTION.get(field) == "desc" else int(val))
    if isinstance(val, (int, float)):
        return (0, -val if _SORT_DIRECTION.get(field, "desc") == "desc" else val)
    return (0, str(val))


def _sort_rows(rows: list[dict], field: str) -> list[dict]:
    return sorted(rows, key=lambda r: _sort_key(r, field))


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

_MD_COLS = [
    ("name",                "Run"),
    ("frontend_type",       "Frontend"),
    ("morlet_use_phase",    "Phase"),
    ("stage2",              "Stage2"),
    ("epochs_completed",    "Ep"),
    ("best_pres_f1",        "pres_f1"),
    ("best_type_f1",        "type_f1"),
    ("best_val_ref_elbo",   "ELBO"),
    ("min_dataset_type_f1", "min-ds F1"),
    ("worst_dataset",       "worst"),
    ("mcc",                 "MCC"),
]


def render_markdown(rows: list[dict], title: str) -> str:
    """Render a sorted row list as GitHub-flavored markdown. Shippable rows
    get a ✓ prepended to the name column."""
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(label for _, label in _MD_COLS) + " |")
    lines.append("|" + "|".join("---" for _ in _MD_COLS) + "|")
    for row in rows:
        cells = []
        for key, _ in _MD_COLS:
            val = row.get(key, "")
            if key == "name" and row.get("shippable"):
                val = f"✓ {val}"
            elif key == "name" and row.get("diverged"):
                val = f"⚠ {val}"
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"- ✓ = shippable (pres_f1 ≥ {A.SHIP_PRES_F1}, type_f1 ≥ {A.SHIP_TYPE_F1})"
    )
    lines.append(f"- ⚠ = diverged (val_ref_elbo > {A.DIVERGED_ELBO_THRESHOLD})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root",    default="saved_crl/runs", type=Path)
    p.add_argument("--out",     default="saved_crl/analysis", type=Path)
    p.add_argument("--filter",  action="append", default=[], metavar="key=val",
                   help="Filter runs by config field (repeatable).")
    p.add_argument("--include-diverged", action="store_true",
                   help="Include runs where best_val_ref_elbo > "
                        f"{A.DIVERGED_ELBO_THRESHOLD} (dropped by default).")
    p.add_argument("--sort", default="best_type_f1",
                   help="Column name to sort by. Direction inferred from "
                        "column semantics (F1s desc, ELBO asc).")
    p.add_argument("--top", type=int, default=None,
                   help="Keep only the top N rows after sorting.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.root)]
    if not runs:
        print(f"No runs found under {args.root}", file=sys.stderr)
        return 1

    filters = A.parse_filter_args(args.filter)
    runs = A.apply_filters(runs, filters, exclude_diverged=not args.include_diverged)

    rows = [_row(rm) for rm in runs]
    rows = _sort_rows(rows, args.sort)
    if args.top is not None:
        rows = rows[:args.top]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "leaderboard.csv"
    md_path  = args.out / "leaderboard.md"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path.write_text(render_markdown(rows, title="CRL Run Leaderboard"))

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote markdown to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
