"""Render the classify shipping bundle's full-split type confusion matrix.

Source data: report.json eval entry for run_name=linear_signal__crl_best_aux_type
on split=full (n=91,325 windows). The aggregator's pre-rendered figure at
top_confusions/type/03_2026-05-03_05-03-14.png uses test-split numbers, which
do not match the full-split prose in the Phase 4 docs. This script renders
the matrix that backs the docs' macro_f1=0.607 / accuracy=0.714 claim.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT = Path(
    "/Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/crl-train/"
    "saved_crl/runs/multiscale/disentangled/2026-05-03_05-03-14/report.json"
)
OUT = Path(
    "/Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/crl-train/"
    "saved_crl/analysis/type_confusion_full_split_2026-05-03_05-03-14"
)
PROBE = "linear_signal__crl_best_aux_type"
CLASS_NAMES = ["pedestrian", "light", "medium", "heavy"]


def main() -> int:
    with REPORT.open() as f:
        r = json.load(f)
    cm = None
    macro_f1 = None
    accuracy = None
    n_windows = None
    for e in r.get("evals", []):
        if e.get("run_name") == PROBE and e.get("split") == "full":
            t = e.get("type") or {}
            if "confusion_matrix" in t:
                cm = np.array(t["confusion_matrix"])
                macro_f1 = t.get("macro_f1")
                accuracy = t.get("accuracy")
                n_windows = e.get("n_windows")
                break
    if cm is None:
        print("ERROR: could not find full-split confusion matrix for", PROBE)
        return 1

    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    pct = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row fraction (recall)", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    title = f"Vehicle type — classify shipping bundle (full split, n={n_windows:,})"
    if macro_f1 is not None:
        title += f"\nmacro F1 = {macro_f1:.3f}"
    if accuracy is not None:
        title += f"  ·  accuracy = {accuracy:.3f}"
    ax.set_title(title, fontsize=10)

    # Annotate cells with count + row %
    for i in range(n):
        rs = row_sums[i, 0] if row_sums[i, 0] > 0 else 1
        for j in range(n):
            count = int(cm[i, j])
            p = 100 * count / rs
            color = "white" if pct[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{count:,}\n({p:.0f}%)",
                ha="center", va="center", color=color, fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(str(OUT) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print(f"wrote {OUT}.png and {OUT}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
