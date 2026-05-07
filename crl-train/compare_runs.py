#!/usr/bin/env python3
"""
compare_runs.py — leaderboard of completed CRL runs.

Walks saved_crl/runs/ (or --root) recursively, reads each run's meta.json +
crl/crl_metrics.csv + downstream/<probe>/downstream_metrics.csv +
eval/<probe>/<head>/<split>/eval_report.json, and emits a sorted leaderboard
as both CSV and markdown. Presence and type metrics live under separate
<head> subdirs so the leaderboard never mixes a pres-best epoch's numbers
with a type-best epoch's.

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
    "best_pres_probe",
    "min_dataset_pres_f1",
    "best_type_f1",
    "best_type_probe",
    "min_dataset_type_f1",
    "worst_dataset",
    "best_val_ref_elbo",
    "calibrated_type_f1",
    "balanced_accuracy",
    "mcc",
    "shippable",
    "diverged",
]


def _short_probe(name: str | None) -> str:
    """Compact a long probe identifier for display.
    'mlp_ztype__crl_best_aux_type' -> 'mlp_ztype_aux'."""
    if not name:
        return ""
    probe, _, ckpt = name.partition("__")
    suffix = "_aux" if ckpt.endswith("aux_type") else ""
    return probe + suffix


def _row(rm: A.RunMetrics) -> dict:
    return {
        "name": rm.name,
        "frontend_type": rm.config.get("frontend_type", ""),
        "morlet_use_phase": rm.config.get("morlet_use_phase", ""),
        "prior_type": rm.config.get("prior_type", ""),
        "training_mode": rm.config.get("training_mode", ""),
        "stage2": rm.stage2,
        "epochs_completed": rm.epochs_completed,
        "best_pres_f1": _fmt(rm.best_pres_f1),
        "best_pres_probe": _short_probe(rm.best_pres_probe),
        "min_dataset_pres_f1": _fmt(rm.min_dataset_pres_f1),
        "best_type_f1": _fmt(rm.best_type_f1),
        "best_type_probe": _short_probe(rm.best_type_probe),
        "min_dataset_type_f1": _fmt(rm.min_dataset_type_f1),
        "worst_dataset": rm.worst_dataset or "",
        "best_val_ref_elbo": _fmt(rm.best_val_ref_elbo),
        "calibrated_type_f1": _fmt(rm.calibrated_type_f1),
        "balanced_accuracy": _fmt(rm.balanced_accuracy),
        "mcc": _fmt(rm.mcc),
        "shippable": rm.shippable,
        "diverged": rm.diverged,
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
    "best_pres_f1": "desc",
    "best_type_f1": "desc",
    "min_dataset_type_f1": "desc",
    "calibrated_type_f1": "desc",
    "balanced_accuracy": "desc",
    "mcc": "desc",
    "shippable": "desc",
    "epochs_completed": "desc",
    "best_val_ref_elbo": "asc",
}


def _sort_key(row: dict, field: str):
    val = row.get(field)
    if val == "" or val is None:
        # Push missing values to the end regardless of direction.
        return (1, 0)
    if isinstance(val, bool):
        return (0, -int(val) if _SORT_DIRECTION.get(field) == "desc" else int(val))
    if isinstance(val, int | float):
        return (0, -val if _SORT_DIRECTION.get(field, "desc") == "desc" else val)
    return (0, str(val))


def _sort_rows(rows: list[dict], field: str) -> list[dict]:
    return sorted(rows, key=lambda r: _sort_key(r, field))


# Primary → tiebreaker pairs that match the bundle-catalog ranking. Used by
# --sort tiebreak-{pres,type} to render the leaderboard in promotion order.
# Detect uses MCC primary (the test set is ~75% positive, so raw F1 rewards
# recall-biased degenerate predictors; MCC is invariant to the prior). The
# `mcc` column on RunMetrics is the test-time presence MCC from the 'full'
# split eval_report.
_TIEBREAK_PAIRS = {
    "tiebreak-pres": ("mcc", "min_dataset_pres_f1"),
    "tiebreak-type": ("best_type_f1", "min_dataset_type_f1"),
}


def _sort_rows_with_tiebreak(rows: list[dict], primary: str, tiebreaker: str) -> list[dict]:
    """Sort rows by `primary` desc, with `tiebreaker` desc as the in-band
    tie-breaker. Inside an ε=A.TIE_EPSILON band the tiebreaker fires;
    outside it the primary alone decides. Mirrors the bundle catalog's
    _rank_bundles in export_for_inference.py.
    """
    def _coerce(v):
        if v == "" or v is None:
            return None
        return float(v) if isinstance(v, int | float) else None

    annotated = []
    for r in rows:
        p = _coerce(r.get(primary))
        t = _coerce(r.get(tiebreaker))
        annotated.append((p, t, r))
    # Push None primaries to the end.
    have, missing = [a for a in annotated if a[0] is not None], [a for a in annotated if a[0] is None]
    have.sort(key=lambda a: (-a[0], a[2].get("name", "")))
    out: list[dict] = []
    i = 0
    while i < len(have):
        anchor = have[i][0]
        end = i
        while end < len(have) and (anchor - have[end][0]) < A.TIE_EPSILON:
            end += 1
        group = have[i:end]
        group.sort(
            key=lambda a: (-(a[1] if a[1] is not None else float("-inf")), a[2].get("name", ""))
        )
        out.extend(a[2] for a in group)
        i = end
    out.extend(a[2] for a in missing)
    return out


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

_MD_COLS = [
    ("name", "Run"),
    ("frontend_type", "Frontend"),
    ("training_mode", "Mode"),
    ("epochs_completed", "Ep"),
    ("mcc", "pres_MCC"),
    ("balanced_accuracy", "pres_BalAcc"),
    ("best_pres_f1", "pres_F1"),
    ("best_pres_probe", "pres_probe"),
    ("min_dataset_pres_f1", "min_pres_F1"),
    ("best_type_f1", "type_F1"),
    ("best_type_probe", "type_probe"),
    ("min_dataset_type_f1", "min_type_F1"),
    ("best_val_ref_elbo", "ELBO"),
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
    lines.append(f"- ✓ = shippable (pres_f1 ≥ {A.SHIP_PRES_F1}, type_f1 ≥ {A.SHIP_TYPE_F1})")
    lines.append(f"- ⚠ = diverged (val_ref_elbo > {A.DIVERGED_ELBO_THRESHOLD})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="saved_crl/runs", type=Path)
    p.add_argument("--out", default="saved_crl/analysis", type=Path)
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="key=val",
        help="Filter runs by config field (repeatable).",
    )
    p.add_argument(
        "--include-diverged",
        action="store_true",
        help="Include runs where best_val_ref_elbo > "
        f"{A.DIVERGED_ELBO_THRESHOLD} (dropped by default).",
    )
    p.add_argument(
        "--sort",
        default="tiebreak-type",
        help="Column name to sort by. Direction inferred from "
        "column semantics (F1s desc, ELBO asc). Special values "
        "'tiebreak-type' and 'tiebreak-pres' apply the bundle-catalog "
        "ranking rule: primary F1 with min-F1 tie-break inside an ε=0.01 "
        "band, so the leaderboard order matches bundle promotion order.",
    )
    p.add_argument("--top", type=int, default=None, help="Keep only the top N rows after sorting.")
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
    if args.sort in _TIEBREAK_PAIRS:
        primary, tiebreaker = _TIEBREAK_PAIRS[args.sort]
        rows = _sort_rows_with_tiebreak(rows, primary, tiebreaker)
    else:
        rows = _sort_rows(rows, args.sort)
    if args.top is not None:
        rows = rows[: args.top]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "leaderboard.csv"
    md_path = args.out / "leaderboard.md"

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
