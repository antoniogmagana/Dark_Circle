"""Generate slide-ready confusion-matrix figure(s) from eval_report.json.

Reads each run's `<run_dir>/eval_report.json` (produced by eval.py) and emits
row-normalized 4x4 vehicle-type confusion matrices, side-by-side.

Soft-exits with a clear message if either run lacks an eval_report.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent  # crl-train/saved_crl/slides-figs/
SAVED_CRL = ROOT.parent  # crl-train/saved_crl/

DEFAULT_MULTI = SAVED_CRL / "id_split" / "multiscale_run1"
DEFAULT_MORLET = SAVED_CRL / "id_split" / "morlet_per_sensor_phase_run1"

LABEL_MULTI = "Multiscale frontend"
LABEL_MORLET = "Morlet (per-sensor) frontend"

CLASS_ORDER = ["pedestrian", "light", "medium", "heavy"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def find_report(run_dir: Path) -> Path | None:
    """Locate the eval_report.json that matches the run's headline probe.

    Resolution order (only 'full' splits are considered — per-dataset
    filtered reports like focal/iobt are skipped):
      1. Top-level <run_dir>/eval_report.json (eval.py default).
      2. <run_dir>/eval/<probe_mode>__<ckpt_stem>/full/eval_report.json,
         where probe_mode and ckpt_name come from meta.json — i.e. the
         same probe×checkpoint that produced the training-curve figures.
      3. Any <run_dir>/eval/**/full/eval_report.json as a last resort.
    """
    top = run_dir / "eval_report.json"
    if top.is_file():
        return top

    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        probe = meta.get("probe_mode")
        ckpt_name = meta.get("ckpt_name")
        if probe and ckpt_name:
            ckpt_stem = Path(ckpt_name).stem
            preferred = (
                run_dir / "eval" / f"{probe}__{ckpt_stem}" / "full" / "eval_report.json"
            )
            if preferred.is_file():
                return preferred

    full_only = sorted(run_dir.glob("eval/**/full/eval_report.json"))
    return full_only[0] if full_only else None


def cm_panel(ax, cm: np.ndarray, title: str) -> None:
    """Plot a row-normalized 4x4 confusion heatmap on the given axes."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")

    n = len(CLASS_ORDER)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=15, pad=10)

    for i in range(n):
        for j in range(n):
            value = norm[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=12
            )
    return im


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--multi-run",
        type=Path,
        default=DEFAULT_MULTI,
        help="Run dir for the multiscale model (default: %(default)s)",
    )
    p.add_argument(
        "--morlet-run",
        type=Path,
        default=DEFAULT_MORLET,
        help="Run dir for the morlet model (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "fig5_confusion_type.png",
        help="Output PNG path",
    )
    args = p.parse_args()

    multi_report = find_report(args.multi_run)
    morlet_report = find_report(args.morlet_run)

    missing = []
    if multi_report is None:
        missing.append(f"  multiscale: {args.multi_run}")
    if morlet_report is None:
        missing.append(f"  morlet:     {args.morlet_run}")
    if missing:
        print("eval_report.json not found for:", *missing, sep="\n")
        print(
            "Run `python eval.py --save-dir <run_dir>` first, or wait for the diagnostic pipeline."
        )
        return 0  # soft exit

    multi = json.loads(multi_report.read_text())
    morlet = json.loads(morlet_report.read_text())

    multi_cm = multi["type"]["confusion_matrix"]
    morlet_cm = morlet["type"]["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.75))
    cm_panel(
        axes[0], multi_cm, f"{LABEL_MULTI}  •  macro F1 {multi['type']['macro_f1']:.3f}"
    )
    im = cm_panel(
        axes[1],
        morlet_cm,
        f"{LABEL_MORLET}  •  macro F1 {morlet['type']['macro_f1']:.3f}",
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, shrink=0.85)
    cbar.set_label("Row-normalized rate (recall view)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  multiscale eval: {multi_report}")
    print(f"  morlet eval:     {morlet_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
