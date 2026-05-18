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

DEFAULT_VAE = SAVED_CRL / "runs" / "multiscale" / "vae" / "2026-05-03_05-02-44"
DEFAULT_DIS = SAVED_CRL / "runs" / "multiscale" / "disentangled" / "2026-05-03_05-03-14"

LABEL_VAE = "Multiscale + VAE"
LABEL_DIS = "Multiscale + Disentangled"

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


def find_report(run_dir: Path, head: str = "type") -> Path | None:
    """Locate the eval_report.json for a given head ("type" or "pres").

    The eval pipeline writes per-head reports at
    `<run>/eval/<probe>__<ckpt>/<head>/full/eval_report.json`. Older runs
    may have a single combined `<run>/eval_report.json` at the top level —
    we use that if present, since it carries both heads.
    """
    top = run_dir / "eval_report.json"
    if top.is_file():
        return top

    # Per-head layout. Look at meta.json first (for crl/<...> nested layouts
    # too), then fall back to alphabetical pick under any probe×ckpt dir.
    meta_paths = (run_dir / "meta.json", run_dir / "crl" / "meta.json")
    for meta_path in meta_paths:
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            probe = meta.get("probe_mode")
            ckpt_name = meta.get("ckpt_name")
            if probe and ckpt_name:
                ckpt_stem = Path(ckpt_name).stem
                preferred = (
                    run_dir / "eval" / f"{probe}__{ckpt_stem}" / head / "full" / "eval_report.json"
                )
                if preferred.is_file():
                    return preferred
            break

    matches = sorted(run_dir.glob(f"eval/*/{head}/full/eval_report.json"))
    return matches[0] if matches else None


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
        "--vae-run",
        type=Path,
        default=DEFAULT_VAE,
        help="Run dir for the multiscale-VAE model (default: %(default)s)",
    )
    p.add_argument(
        "--dis-run",
        type=Path,
        default=DEFAULT_DIS,
        help="Run dir for the multiscale-Disentangled model (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "fig5_confusion_type.png",
        help="Output PNG path",
    )
    args = p.parse_args()

    vae_report = find_report(args.vae_run, head="type")
    dis_report = find_report(args.dis_run, head="type")

    missing = []
    if vae_report is None:
        missing.append(f"  vae:           {args.vae_run}")
    if dis_report is None:
        missing.append(f"  disentangled:  {args.dis_run}")
    if missing:
        print("eval_report.json not found for:", *missing, sep="\n")
        print(
            "Run `python eval.py --save-dir <run_dir>` first, or wait for the diagnostic pipeline."
        )
        return 0  # soft exit

    vae = json.loads(vae_report.read_text())
    dis = json.loads(dis_report.read_text())

    vae_cm = vae["type"]["confusion_matrix"]
    dis_cm = dis["type"]["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.75))
    cm_panel(
        axes[0], vae_cm, f"{LABEL_VAE}  •  macro F1 {vae['type']['macro_f1']:.3f}"
    )
    im = cm_panel(
        axes[1],
        dis_cm,
        f"{LABEL_DIS}  •  macro F1 {dis['type']['macro_f1']:.3f}",
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, shrink=0.85)
    cbar.set_label("Row-normalized rate (recall view)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  vae eval:           {vae_report}")
    print(f"  disentangled eval:  {dis_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
