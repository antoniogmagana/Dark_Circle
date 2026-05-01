"""Generate slide-ready figures comparing multiscale vs morlet_per_sensor base CRL models.

Outputs PNGs at 300 DPI, 16:9, viridis-friendly palette, no titles.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # crl-train/saved_crl/slides-figs/
SAVED_CRL = ROOT.parent  # crl-train/saved_crl/
DEFAULT_MULTI = SAVED_CRL / "id_split" / "multiscale_run1"
DEFAULT_MORLET = SAVED_CRL / "id_split" / "morlet_per_sensor_phase_run1"

# Color-blind safe pair drawn from viridis endpoints (dark purple, teal-green).
COLOR_MULTI = "#440154"
COLOR_MORLET = "#21918c"

LABEL_MULTI = "Multiscale frontend"
LABEL_MORLET = "Morlet (per-sensor) frontend"

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
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

FIGSIZE_16_9 = (12.0, 6.75)


def load_run(path: Path) -> dict:
    crl = pd.read_csv(path / "crl_metrics.csv")
    down = pd.read_csv(path / "downstream_metrics.csv")
    with (path / "crl_checkpoint_summary.json").open() as f:
        ckpt = json.load(f)
    with (path / "meta.json").open() as f:
        meta = json.load(f)
    return {"crl": crl, "down": down, "ckpt": ckpt, "meta": meta}


def fig1_downstream_f1_bars(multi: dict, morlet: dict, out: Path) -> None:
    """Best downstream val F1 (presence + type) per model — grouped bars."""
    metrics = ["val_pres_f1", "val_type_f1"]
    pretty = ["Presence F1", "Vehicle-type F1"]
    multi_best = [multi["down"][m].max() for m in metrics]
    morlet_best = [morlet["down"][m].max() for m in metrics]

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    x = np.arange(len(metrics))
    width = 0.35
    b1 = ax.bar(x - width / 2, multi_best, width, label=LABEL_MULTI, color=COLOR_MULTI)
    b2 = ax.bar(
        x + width / 2, morlet_best, width, label=LABEL_MORLET, color=COLOR_MORLET
    )
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("Best validation F1 (downstream probe)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2_downstream_typef1_curves(multi: dict, morlet: dict, out: Path) -> None:
    """Validation type F1 across downstream training epochs."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    ax.plot(
        multi["down"]["epoch"],
        multi["down"]["val_type_f1"],
        color=COLOR_MULTI,
        label=LABEL_MULTI,
    )
    ax.plot(
        morlet["down"]["epoch"],
        morlet["down"]["val_type_f1"],
        color=COLOR_MORLET,
        label=LABEL_MORLET,
    )
    ax.set_xlabel("Downstream probe epoch")
    ax.set_ylabel("Validation vehicle-type F1")
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_ref_elbo(multi: dict, morlet: dict, out: Path) -> None:
    """Validation reference ELBO across CRL pretraining epochs (β-invariant)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    ax.plot(
        multi["crl"]["epoch"],
        multi["crl"]["val_ref_elbo"],
        color=COLOR_MULTI,
        label=LABEL_MULTI,
    )
    ax.plot(
        morlet["crl"]["epoch"],
        morlet["crl"]["val_ref_elbo"],
        color=COLOR_MORLET,
        label=LABEL_MORLET,
    )
    ax.set_xlabel("CRL pretraining epoch")
    ax.set_ylabel("Validation reference ELBO  (recon + raw KL at β=1)")
    ax.set_yscale("log")
    ax.yaxis.grid(True, which="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig4_aux_type_f1(multi: dict, morlet: dict, out: Path) -> None:
    """Auxiliary type-classifier F1 during CRL pretraining (proxy signal)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)

    ax.plot(
        multi["crl"]["epoch"],
        multi["crl"]["val_aux_type_f1"],
        color=COLOR_MULTI,
        label=LABEL_MULTI,
    )
    ax.plot(
        morlet["crl"]["epoch"],
        morlet["crl"]["val_aux_type_f1"],
        color=COLOR_MORLET,
        label=LABEL_MORLET,
    )

    for run, color in ((multi, COLOR_MULTI), (morlet, COLOR_MORLET)):
        best_ep = int(run["ckpt"]["best_aux_type_epoch"])
        best_v = float(run["ckpt"]["best_aux_type_f1"])
        ax.scatter(
            [best_ep],
            [best_v],
            color=color,
            s=80,
            zorder=5,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            f"best  ep {best_ep}, F1 {best_v:.3f}",
            xy=(best_ep, best_v),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
            color=color,
        )

    ax.set_xlabel("CRL pretraining epoch")
    ax.set_ylabel("Validation aux type F1  (proxy probe)")
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
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
    args = p.parse_args()

    multi = load_run(args.multi_run)
    morlet = load_run(args.morlet_run)

    fig1_downstream_f1_bars(multi, morlet, ROOT / "fig1_downstream_f1_bars.png")
    fig2_downstream_typef1_curves(
        multi, morlet, ROOT / "fig2_downstream_type_f1_curves.png"
    )
    fig3_ref_elbo(multi, morlet, ROOT / "fig3_crl_val_ref_elbo.png")
    fig4_aux_type_f1(multi, morlet, ROOT / "fig4_crl_aux_type_f1.png")

    summary = {
        "multiscale": {
            "downstream_best_pres_f1": float(multi["down"]["val_pres_f1"].max()),
            "downstream_best_type_f1": float(multi["down"]["val_type_f1"].max()),
            "downstream_best_type_f1_epoch": int(multi["down"]["val_type_f1"].idxmax()),
            "crl_best_ref_elbo": multi["ckpt"]["best_ref_elbo"],
            "crl_best_aux_type_f1": multi["ckpt"]["best_aux_type_f1"],
            "crl_best_aux_type_epoch": multi["ckpt"]["best_aux_type_epoch"],
            "crl_epochs_run": int(multi["crl"]["epoch"].max()) + 1,
            "down_epochs_run": int(multi["down"]["epoch"].max()) + 1,
        },
        "morlet_per_sensor": {
            "downstream_best_pres_f1": float(morlet["down"]["val_pres_f1"].max()),
            "downstream_best_type_f1": float(morlet["down"]["val_type_f1"].max()),
            "downstream_best_type_f1_epoch": int(
                morlet["down"]["val_type_f1"].idxmax()
            ),
            "crl_best_ref_elbo": morlet["ckpt"]["best_ref_elbo"],
            "crl_best_aux_type_f1": morlet["ckpt"]["best_aux_type_f1"],
            "crl_best_aux_type_epoch": morlet["ckpt"]["best_aux_type_epoch"],
            "crl_epochs_run": int(morlet["crl"]["epoch"].max()) + 1,
            "down_epochs_run": int(morlet["down"]["epoch"].max()) + 1,
        },
    }
    (ROOT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
