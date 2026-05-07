#!/usr/bin/env python3
"""
gather_top_confusions.py — re-render confusion matrices for top performers.

Ranks runs the same way the bundle catalog promotes defaults: primary val
F1 (val_pres_f1 / val_type_f1) with min-F1 tie-breaker inside an ε=0.01
band. Each run's per-head winning probe drives both the ranking and the
confusion matrix plotted. This means the top-1 PNG is from the same run
+ probe combination that would win `--promote-default` if exported now.

Output paths:

    saved_crl/analysis/top_confusions/detection/<rank>_<run-name>.{png,pdf}
    saved_crl/analysis/top_confusions/type/<rank>_<run-name>.{png,pdf}

The confusion-matrix data itself comes from eval_report.json (test-time
inference results — `presence.{tp,fp,fn,tn}` for binary,
`type.confusion_matrix` for multiclass). We don't re-run inference.

Pass `--probe` to force a single probe across all runs (debugging /
poster overrides). Default is auto-select per the bundle catalog rule.

Usage
-----
    python gather_top_confusions.py
    python gather_top_confusions.py --top-n 3
    python gather_top_confusions.py --probe linear_signal__crl_best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A
from crl_vehicle import plotting as P


def find_eval_report(run_dir: Path, probe: str, head: str | None = None) -> Path | None:
    """Locate eval_report.json for the `full` test split.

    Two on-disk layouts coexist:
        eval/<probe>/full/eval_report.json              (older)
        eval/<probe>/<head>/full/eval_report.json       (per-head, newer)

    `head` is optional; when set, only the per-head layout is searched
    under that head's subdir. Falls back to any other probe under eval/
    if the canonical probe is missing.
    """
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return None
    candidate_probes = [eval_root / probe]
    candidate_probes.extend(
        sorted(p for p in eval_root.iterdir() if p.is_dir() and p.name != probe)
    )
    for probe_dir in candidate_probes:
        if not probe_dir.is_dir():
            continue
        if head is not None:
            per_head = probe_dir / head / "full" / "eval_report.json"
            if per_head.exists():
                return per_head
            continue
        flat = probe_dir / "full" / "eval_report.json"
        if flat.exists():
            return flat
        # Fall back to per-head layout, either head, when caller didn't pin one.
        for h in ("type", "pres"):
            ph = probe_dir / h / "full" / "eval_report.json"
            if ph.exists():
                return ph
    return None


def render_presence(report: dict, run_name: str, out_stem: Path) -> bool:
    """Render binary presence confusion. Returns True if written."""
    pres = report.get("presence")
    if not pres or "tp" not in pres:
        return False
    tn, fp, fn, tp = pres["tn"], pres["fp"], pres["fn"], pres["tp"]
    cm = [[tn, fp], [fn, tp]]
    for ext in (".png", ".pdf"):
        P.plot_confusion_matrix(
            cm,
            ["absent", "present"],
            run_name,
            out_stem.with_suffix(ext),
            subtitle="Presence (test)",
        )
    print(f"  wrote {out_stem.with_suffix('.png')}")
    return True


def render_type(report: dict, run_name: str, out_stem: Path) -> bool:
    """Render multiclass type confusion. Returns True if written."""
    type_block = report.get("type")
    if not type_block or "confusion_matrix" not in type_block:
        return False
    cm = type_block["confusion_matrix"]
    # Class names from per_class keys (insertion order = class index).
    per_class = type_block.get("per_class", {})
    class_names = list(per_class.keys()) if per_class else [str(i) for i in range(len(cm))]
    for ext in (".png", ".pdf"):
        P.plot_confusion_matrix(
            cm,
            class_names,
            run_name,
            out_stem.with_suffix(ext),
            subtitle="Vehicle type (test)",
        )
    print(f"  wrote {out_stem.with_suffix('.png')}")
    return True


def _rank_runs_with_tiebreak(
    runs: list[A.RunMetrics], head: str
) -> list[A.RunMetrics]:
    """Rank runs the same way the bundle catalog does (see
    export_for_inference.py:_rank_bundles + _RANKING_METRICS): MCC primary
    for presence (the test set is ~75% positive so raw F1 rewards recall-
    biased predictors), val type-F1 primary for type, with the per-head
    min-F1 as tie-breaker inside an ε=TIE_EPSILON band. Each run
    contributes its per-head winner, so the same probe selection that
    wins promotion is the one whose confusion matrix gets plotted.
    """
    primary_attr = "mcc" if head == "presence" else "best_type_f1"
    min_attr = "min_dataset_pres_f1" if head == "presence" else "min_dataset_type_f1"
    candidates: list[tuple[float, float, str, A.RunMetrics]] = []
    for rm in runs:
        primary = getattr(rm, primary_attr)
        if primary is None:
            continue
        tiebreaker = getattr(rm, min_attr)
        # Sort key: (-primary, -tiebreaker, name) so primary descending,
        # then tie-break descending, then name ascending for determinism.
        # We bin by ε on the primary axis below.
        candidates.append(
            (primary, tiebreaker if tiebreaker is not None else float("-inf"), rm.name, rm)
        )
    if not candidates:
        return []
    candidates.sort(key=lambda c: (-c[0], c[2]))
    out: list[A.RunMetrics] = []
    i = 0
    while i < len(candidates):
        anchor = candidates[i][0]
        end = i
        while end < len(candidates) and (anchor - candidates[end][0]) < A.TIE_EPSILON:
            end += 1
        group = candidates[i:end]
        group.sort(key=lambda c: (-c[1], c[2]))
        out.extend(c[3] for c in group)
        i = end
    return out


def gather(
    runs: list[A.RunMetrics],
    head: str,
    out_subdir: Path,
    probe: str | None,
    top_n: int,
) -> None:
    """Pick top-N runs by val F1 with min-F1 tie-breaker (matching the
    bundle-catalog promotion rule). For each winner, render its test-time
    confusion matrix from the per-head winning probe.

    `head` is one of {"presence", "type"} — drives both the metric used
    for ranking (val_pres_f1 vs val_type_f1) and which block of the
    report gets plotted.
    """
    ranked = _rank_runs_with_tiebreak(runs, head)
    if not ranked:
        print(f"  no runs with val {head} F1; skipping")
        return
    top = ranked[:top_n]
    head_subdir = "pres" if head == "presence" else "type"
    per_head_field = "best_pres_probe" if head == "presence" else "best_type_probe"
    primary_attr = "mcc" if head == "presence" else "best_type_f1"
    min_attr = "min_dataset_pres_f1" if head == "presence" else "min_dataset_type_f1"
    metric_label = "pres_MCC" if head == "presence" else "val_type_f1"
    print(f"Top {len(top)} by {metric_label} (with min-F1 tie-breaker):")
    for rank, rm in enumerate(top, start=1):
        per_run_probe = probe or getattr(rm, per_head_field, None) or A.CANONICAL_PROBE
        report_path = find_eval_report(rm.path, per_run_probe, head=head_subdir) or find_eval_report(
            rm.path, per_run_probe, head=None
        )
        primary = getattr(rm, primary_attr)
        tb = getattr(rm, min_attr)
        tb_str = f"{tb:.3f}" if tb is not None else "—"
        provenance = (
            f"{report_path.relative_to(rm.path)}" if report_path is not None else "(no eval_report.json)"
        )
        print(
            f"  {rank}. {rm.name}  {metric_label}={primary:.3f}  "
            f"min_f1={tb_str}  probe={per_run_probe}  "
            f"({rm.config.get('frontend_type','?')})  ← {provenance}"
        )
        if report_path is None:
            continue
        report = json.loads(report_path.read_text())
        out_stem = out_subdir / f"{rank:02d}_{rm.name}"
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        ok = (
            render_presence(report, rm.name, out_stem)
            if head == "presence"
            else render_type(report, rm.name, out_stem)
        )
        if not ok:
            print(f"     (eval_report.json missing '{head}' block at {report_path})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="saved_crl/runs", type=Path)
    p.add_argument("--out", default="saved_crl/analysis/top_confusions", type=Path)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument(
        "--probe",
        default=None,
        help="Probe to pull eval_report from. Default behavior is per-head "
        "auto-select: presence head reads from each run's best_pres_probe, "
        "type head reads from each run's best_type_probe. Pass an explicit "
        "probe name (e.g. linear_signal__crl_best) to override and force a "
        "single probe across all runs and both heads.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    P.apply_poster_style()
    runs = [A.load_run_metrics(p) for p in A.discover_runs(args.root)]
    if not runs:
        print(f"No runs found under {args.root}", file=sys.stderr)
        return 1
    runs = A.apply_filters(runs, {}, exclude_diverged=True)

    gather(runs, "presence", args.out / "detection", args.probe, args.top_n)
    gather(runs, "type", args.out / "type", args.probe, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
