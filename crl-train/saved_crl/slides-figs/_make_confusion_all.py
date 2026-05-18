"""Confusion matrices for all completed id_split runs (one panel per run).

Reads each run's headline probe×ckpt eval_report.json (resolved via the
run's meta.json) and emits two figures:

  fig13_confusion_type_all_runs.png      — 4x4 vehicle-type CM per run
  fig14_confusion_presence_all_runs.png  — 2x2 presence (binary) CM per run

Both figures are row-normalized (recall view). Soft-exits if any run is
missing an eval — won't half-render. Override defaults with --runs.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SAVED_CRL = ROOT.parent
RUNS = SAVED_CRL / "runs"

# Defaults: the two multiscale-only runs in the comparison set
# (Multiscale + VAE vs Multiscale + Disentangled).
DEFAULT_RUNS = [
    RUNS / "multiscale" / "vae" / "2026-05-03_05-02-44",
    RUNS / "multiscale" / "disentangled" / "2026-05-03_05-03-14",
]

# Shipping configuration matching the paper's Performance Evaluation table.
# Each task points at one (run, probe, ckpt) triple.
SHIPPING = {
    "presence": {
        "run": RUNS / "multiscale" / "vae" / "2026-05-03_05-02-44",
        "probe": "mlp_ztype",
        "ckpt": "crl_best.pth",
        "label": "VAE 2026-05-03_05-02-44 · mlp_ztype × crl_best",
    },
    "type": {
        "run": RUNS / "multiscale" / "disentangled" / "2026-05-03_05-03-14",
        "probe": "linear_signal",
        "ckpt": "crl_best_aux_type.pth",
        "label": "DIS 2026-05-03_05-03-14 · linear_signal × crl_best_aux_type",
    },
}

CLASS_ORDER = ["pedestrian", "light", "medium", "heavy"]
PRESENCE_LABELS = ["no vehicle", "vehicle"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def headline_probe_ckpt(run_dir: Path) -> tuple[str | None, str | None]:
    """Return (probe_mode, ckpt_name) the run was trained against.

    Looks at <run>/meta.json first; if absent (full-diagnostic-pipeline
    layout where meta lives under crl/meta.json), falls back to that.
    """
    for cand in (run_dir / "meta.json", run_dir / "crl" / "meta.json"):
        if cand.is_file():
            m = json.loads(cand.read_text())
            return m.get("probe_mode"), m.get("ckpt_name")
    return None, None


def find_eval_report(run_dir: Path, head: str = "type") -> Path | None:
    """Locate the headline eval_report.json for a run, for a given head.

    `head` is "type" or "pres" — the eval pipeline writes per-head reports at
    `<run>/eval/<probe>__<ckpt>/<head>/full/eval_report.json`.

    Resolution order, only 'full' splits considered:
      1. Top-level <run>/eval_report.json (plain `eval.py` output, combined).
      2. <run>/eval/<probe>__<ckpt_stem>/<head>/full/eval_report.json from meta.
      3. Any <run>/eval/*/<head>/full/eval_report.json (alphabetical).
    """
    top = run_dir / "eval_report.json"
    if top.is_file():
        return top
    probe, ckpt_name = headline_probe_ckpt(run_dir)
    if probe and ckpt_name:
        ckpt_stem = Path(ckpt_name).stem
        preferred = (
            run_dir / "eval" / f"{probe}__{ckpt_stem}" / head / "full" / "eval_report.json"
        )
        if preferred.is_file():
            return preferred
    matches = sorted(run_dir.glob(f"eval/*/{head}/full/eval_report.json"))
    return matches[0] if matches else None


def cm_panel(
    ax, cm, title: str, labels, *, show_y: bool = True, value_fontsize: int = 11
):
    """Plot a row-normalized confusion heatmap."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Predicted class")
    if show_y:
        ax.set_yticklabels(labels)
        ax.set_ylabel("True class")
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.set_title(title, fontsize=14, pad=10)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.5 else "black",
                fontsize=value_fontsize,
            )
    return im


def presence_cm(rep: dict) -> list[list[int]]:
    """Build a 2x2 presence confusion matrix from the eval_report counts.

    Layout (row=true, col=predicted), matching the type CM convention:
      [[tn, fp],
       [fn, tp]]
    """
    p = rep["presence"]
    return [[int(p["tn"]), int(p["fp"])], [int(p["fn"]), int(p["tp"])]]


def render_grid(reports, kind: str, out: Path) -> None:
    """Render a row of confusion-matrix panels — one per run.

    kind: "type" or "presence". Selects which CM and labels to use.
    """
    if kind == "type":
        labels = CLASS_ORDER
        value_fs = 11
    elif kind == "presence":
        labels = PRESENCE_LABELS
        value_fs = 14
    else:
        raise ValueError(kind)

    # Single panel size for both kinds so the type and presence figures match
    # when placed side-by-side in the paper.
    panel_w = 5.5

    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n + 1.5, 6.0))
    if n == 1:
        axes = [axes]

    last_im = None
    for i, (run, rep_path, rep) in enumerate(reports):
        if kind == "type":
            cm = rep["type"]["confusion_matrix"]
            # Type evaluation runs on vehicle-positive windows only; the
            # eval_report's top-level n_windows is the paired-presence count,
            # which would mislead readers. Sum the matrix instead.
            nw = int(sum(sum(row) for row in cm))
            score_label = f"macro F1 {rep['type']['macro_f1']:.3f}"
            n_label = f"n={nw:,} vehicle windows"
        else:
            cm = presence_cm(rep)
            nw = int(sum(sum(row) for row in cm))
            n_label = f"n={nw:,}"
            score_label = f"F1 {rep['presence']['f1']:.3f}"
        title = (
            f"{run.name}\n"
            f"{rep['probe_mode']} · {rep['ckpt_name']}\n"
            f"{score_label}  ·  {n_label}"
        )
        last_im = cm_panel(
            axes[i], cm, title, labels, show_y=(i == 0), value_fontsize=value_fs
        )

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.022, pad=0.04, shrink=0.85)
    cbar.set_label("Row-normalized rate (recall view)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(out)
    plt.close(fig)


def shipping_report_path(task: str) -> Path:
    """Resolve the eval_report.json for the shipping (run, probe, ckpt) triple."""
    cfg = SHIPPING[task]
    head_dir = "type" if task == "type" else "pres"
    ckpt_stem = Path(cfg["ckpt"]).stem
    return (
        cfg["run"]
        / "eval"
        / f"{cfg['probe']}__{ckpt_stem}"
        / head_dir
        / "full"
        / "eval_report.json"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shipping",
        action="store_true",
        help=(
            "Render single-panel-per-task figures using the SHIPPING config "
            "(VAE × mlp_ztype × crl_best for detection; "
            "Disentangled × linear_signal × crl_best_aux_type for type). "
            "Default mode is the multi-run grid using --runs."
        ),
    )
    p.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run dirs to render in grid mode (ignored when --shipping)",
    )
    p.add_argument(
        "--out-type",
        type=Path,
        default=ROOT / "fig13_confusion_type_all_runs.png",
        help="Output PNG path for the type confusion figure",
    )
    p.add_argument(
        "--out-presence",
        type=Path,
        default=ROOT / "fig14_confusion_presence_all_runs.png",
        help="Output PNG path for the presence confusion figure",
    )
    args = p.parse_args()

    if args.shipping:
        # Single-panel-per-task: each task points at one (run, probe, ckpt) triple.
        type_path = shipping_report_path("type")
        pres_path = shipping_report_path("presence")
        missing = [str(p) for p in (type_path, pres_path) if not p.is_file()]
        if missing:
            print("Skipping — shipping eval_report.json not found:")
            for m in missing:
                print(f"  {m}")
            return 0

        type_run = SHIPPING["type"]["run"]
        pres_run = SHIPPING["presence"]["run"]
        type_reports = [(type_run, type_path, json.loads(type_path.read_text()))]
        pres_reports = [(pres_run, pres_path, json.loads(pres_path.read_text()))]

        render_grid(type_reports, "type", args.out_type)
        render_grid(pres_reports, "presence", args.out_presence)
        print(f"wrote {args.out_type}")
        print(f"wrote {args.out_presence}")
        print(f"  type · {type_run.name}: {type_path.relative_to(SAVED_CRL)}")
        print(f"  pres · {pres_run.name}: {pres_path.relative_to(SAVED_CRL)}")
        return 0

    # Default: grid mode over --runs (kept for non-shipping comparison plots).
    def collect(head: str) -> tuple[list[tuple[Path, Path, dict]], list[str]]:
        reports = []
        missing = []
        for run in args.runs:
            rep_path = find_eval_report(run, head=head)
            if rep_path is None:
                missing.append(f"{run}  ({head})")
                continue
            reports.append((run, rep_path, json.loads(rep_path.read_text())))
        return reports, missing

    type_reports, type_missing = collect("type")
    pres_reports, pres_missing = collect("pres")

    if type_missing or pres_missing:
        print("Skipping — eval_report.json not found for:")
        for m in type_missing + pres_missing:
            print(f"  {m}")
        print("Run completes its eval phase first, or pass --runs explicitly.")
        return 0

    if not type_reports or not pres_reports:
        print("No usable runs found.")
        return 0

    render_grid(type_reports, "type", args.out_type)
    render_grid(pres_reports, "presence", args.out_presence)
    print(f"wrote {args.out_type}")
    print(f"wrote {args.out_presence}")
    for run, rep_path, _ in type_reports:
        print(f"  type · {run.name}: {rep_path.relative_to(SAVED_CRL)}")
    for run, rep_path, _ in pres_reports:
        print(f"  pres · {run.name}: {rep_path.relative_to(SAVED_CRL)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
