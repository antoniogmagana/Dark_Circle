#!/usr/bin/env python3
"""
gather_top_confusions.py — re-render confusion matrices for top performers.

Picks the top-N runs twice — once by **test-time** presence F1 and once by
**test-time** type macro_f1 — and re-renders each winner's confusion matrix
at poster styling from the saved `eval_report.json`. Output paths:

    saved_crl/analysis/top_confusions/detection/<rank>_<run-name>.{png,pdf}
    saved_crl/analysis/top_confusions/type/<rank>_<run-name>.{png,pdf}

The raw confusion data and per-head test F1 are already on disk in
eval_report.json (`presence.f1` for binary, `type.macro_f1` for
multiclass, plus `presence.{tp,fp,fn,tn}` and `type.confusion_matrix`).
We don't need to re-run inference — just reload + re-plot through the
shared `P.plot_confusion_matrix()` so the new poster rcParams apply.

Ranking uses test-time F1 from the report itself, *not* the val-time
`best_pres_f1` / `best_type_f1` cached on RunMetrics — that way the
poster plots are ranked by the same number a reader sees on the figure.

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
    P.plot_confusion_matrix(
        cm,
        ["absent", "present"],
        f"{run_name} — Presence (test)",
        out_stem.with_suffix(".png"),
    )
    P.plot_confusion_matrix(
        cm,
        ["absent", "present"],
        f"{run_name} — Presence (test)",
        out_stem.with_suffix(".pdf"),
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
    P.plot_confusion_matrix(
        cm,
        class_names,
        f"{run_name} — Vehicle type (test)",
        out_stem.with_suffix(".png"),
    )
    P.plot_confusion_matrix(
        cm,
        class_names,
        f"{run_name} — Vehicle type (test)",
        out_stem.with_suffix(".pdf"),
    )
    print(f"  wrote {out_stem.with_suffix('.png')}")
    return True


def _test_f1(report: dict, head: str) -> float | None:
    """Pull test-time F1 from an eval_report. presence → presence.f1
    (binary), type → type.macro_f1."""
    if head == "presence":
        block = report.get("presence") or {}
        return block.get("f1")
    block = report.get("type") or {}
    return block.get("macro_f1")


def _score_runs(
    runs: list[A.RunMetrics], head: str, probe: str
) -> list[tuple[float, A.RunMetrics, Path, dict]]:
    """For each run, locate its eval_report.json and pull test F1 for
    `head`. Returns [(score, rm, report_path, report_dict)] for runs
    where both the report and the test-F1 field exist."""
    head_subdir = "pres" if head == "presence" else "type"
    scored: list[tuple[float, A.RunMetrics, Path, dict]] = []
    for rm in runs:
        report_path = find_eval_report(rm.path, probe, head=head_subdir) or find_eval_report(
            rm.path, probe, head=None
        )
        if report_path is None:
            continue
        report = json.loads(report_path.read_text())
        score = _test_f1(report, head)
        if score is None:
            continue
        scored.append((score, rm, report_path, report))
    return scored


def gather(
    runs: list[A.RunMetrics],
    head: str,
    out_subdir: Path,
    probe: str,
    top_n: int,
) -> None:
    """Pick top-N runs by **test-time** F1 for `head`, re-render each
    winner's confusion at poster styling.

    `head` is one of {"presence", "type"} — drives both the metric used
    for ranking (presence.f1 vs type.macro_f1) and which block of the
    report gets plotted.
    """
    scored = _score_runs(runs, head, probe)
    if not scored:
        print(f"  no runs with test-time {head} F1; skipping")
        return
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:top_n]
    metric_label = "presence.f1" if head == "presence" else "type.macro_f1"
    print(f"Top {len(top)} by test-time {metric_label}:")
    for rank, (score, rm, report_path, report) in enumerate(top, start=1):
        print(
            f"  {rank}. {rm.name}  {metric_label}={score:.3f}  "
            f"({rm.config.get('frontend_type','?')})  ← {report_path.relative_to(rm.path)}"
        )
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
        default=A.CANONICAL_PROBE,
        help=f"Probe to pull eval_report from (default: {A.CANONICAL_PROBE}). "
        "Falls back to any available probe per run if missing.",
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
