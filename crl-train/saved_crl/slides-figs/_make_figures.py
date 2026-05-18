"""Generate slide-ready figures comparing multiscale-VAE vs multiscale-Disentangled.

Both runs share the multiscale frontend; the only science variable is the
training objective (VAE vs Disentangled). Both use the same downstream probe
variant `linear_fullz__crl_best` so probe choice does not confound the
comparison.

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
DEFAULT_VAE = SAVED_CRL / "runs" / "multiscale" / "vae" / "2026-05-03_05-02-44"
DEFAULT_DIS = SAVED_CRL / "runs" / "multiscale" / "disentangled" / "2026-05-03_05-03-14"
PROBE_VARIANT = "linear_fullz__crl_best"

# Color-blind safe pair drawn from viridis endpoints (dark purple, teal-green).
COLOR_VAE = "#440154"
COLOR_DIS = "#21918c"

LABEL_VAE = "Multiscale + VAE"
LABEL_DIS = "Multiscale + Disentangled"

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


def load_run(run_dir: Path, probe_variant: str = PROBE_VARIANT) -> dict:
    """Load CRL + downstream + checkpoint summary for one run.

    Expects the layout produced by Trainer.train_crl + train_downstream:
        run_dir/crl/crl_metrics.csv
        run_dir/crl/crl_checkpoint_summary.json
        run_dir/crl/meta.json
        run_dir/downstream/<probe_variant>/downstream_metrics.csv
    """
    crl = pd.read_csv(run_dir / "crl" / "crl_metrics.csv")
    down = pd.read_csv(run_dir / "downstream" / probe_variant / "downstream_metrics.csv")
    with (run_dir / "crl" / "crl_checkpoint_summary.json").open() as f:
        ckpt = json.load(f)
    with (run_dir / "crl" / "meta.json").open() as f:
        meta = json.load(f)
    return {"crl": crl, "down": down, "ckpt": ckpt, "meta": meta}


def fig1_downstream_f1_bars(vae: dict, dis: dict, out: Path) -> None:
    """Best downstream val F1 (presence + type) per training mode — grouped bars."""
    metrics = ["val_pres_f1", "val_type_f1"]
    pretty = ["Presence F1", "Vehicle-type F1"]
    vae_best = [vae["down"][m].max() for m in metrics]
    dis_best = [dis["down"][m].max() for m in metrics]

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    x = np.arange(len(metrics))
    width = 0.35
    b1 = ax.bar(x - width / 2, vae_best, width, label=LABEL_VAE, color=COLOR_VAE)
    b2 = ax.bar(x + width / 2, dis_best, width, label=LABEL_DIS, color=COLOR_DIS)
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
    ax.set_ylabel("Best validation F1 (downstream linear_fullz probe)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2_downstream_typef1_curves(vae: dict, dis: dict, out: Path) -> None:
    """Validation type F1 across downstream training epochs."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    ax.plot(vae["down"]["epoch"], vae["down"]["val_type_f1"], color=COLOR_VAE, label=LABEL_VAE)
    ax.plot(dis["down"]["epoch"], dis["down"]["val_type_f1"], color=COLOR_DIS, label=LABEL_DIS)
    ax.set_xlabel("Downstream probe epoch")
    ax.set_ylabel("Validation vehicle-type F1")
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_ref_elbo(vae: dict, dis: dict, out: Path) -> None:
    """Validation reference ELBO across CRL pretraining epochs (β-invariant)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    ax.plot(vae["crl"]["epoch"], vae["crl"]["val_ref_elbo"], color=COLOR_VAE, label=LABEL_VAE)
    ax.plot(dis["crl"]["epoch"], dis["crl"]["val_ref_elbo"], color=COLOR_DIS, label=LABEL_DIS)
    ax.set_xlabel("CRL pretraining epoch")
    ax.set_ylabel("Validation reference ELBO  (recon + raw KL at β=1)")
    ax.set_yscale("log")
    ax.yaxis.grid(True, which="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig4_aux_type_f1(vae: dict, dis: dict, out: Path) -> None:
    """Auxiliary type-classifier F1 during CRL pretraining (proxy signal)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9)
    ax.plot(vae["crl"]["epoch"], vae["crl"]["val_aux_type_f1"], color=COLOR_VAE, label=LABEL_VAE)
    ax.plot(dis["crl"]["epoch"], dis["crl"]["val_aux_type_f1"], color=COLOR_DIS, label=LABEL_DIS)

    for run, color in ((vae, COLOR_VAE), (dis, COLOR_DIS)):
        best_ep = int(run["ckpt"]["best_aux_type_epoch"])
        best_v = float(run["ckpt"]["best_aux_type_f1"])
        ax.scatter(
            [best_ep], [best_v], color=color, s=80, zorder=5, edgecolor="white", linewidth=1.5
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
    args = p.parse_args()

    vae = load_run(args.vae_run)
    dis = load_run(args.dis_run)

    fig1_downstream_f1_bars(vae, dis, ROOT / "fig1_downstream_f1_bars.png")
    fig2_downstream_typef1_curves(vae, dis, ROOT / "fig2_downstream_type_f1_curves.png")
    fig3_ref_elbo(vae, dis, ROOT / "fig3_crl_val_ref_elbo.png")
    fig4_aux_type_f1(vae, dis, ROOT / "fig4_crl_aux_type_f1.png")

    summary = {
        "multiscale_vae": {
            "downstream_best_pres_f1": float(vae["down"]["val_pres_f1"].max()),
            "downstream_best_type_f1": float(vae["down"]["val_type_f1"].max()),
            "downstream_best_type_f1_epoch": int(vae["down"]["val_type_f1"].idxmax()),
            "crl_best_ref_elbo": vae["ckpt"]["best_ref_elbo"],
            "crl_best_aux_type_f1": vae["ckpt"]["best_aux_type_f1"],
            "crl_best_aux_type_epoch": vae["ckpt"]["best_aux_type_epoch"],
            "crl_epochs_run": int(vae["crl"]["epoch"].max()) + 1,
            "down_epochs_run": int(vae["down"]["epoch"].max()) + 1,
        },
        "multiscale_disentangled": {
            "downstream_best_pres_f1": float(dis["down"]["val_pres_f1"].max()),
            "downstream_best_type_f1": float(dis["down"]["val_type_f1"].max()),
            "downstream_best_type_f1_epoch": int(dis["down"]["val_type_f1"].idxmax()),
            "crl_best_ref_elbo": dis["ckpt"]["best_ref_elbo"],
            "crl_best_aux_type_f1": dis["ckpt"]["best_aux_type_f1"],
            "crl_best_aux_type_epoch": dis["ckpt"]["best_aux_type_epoch"],
            "crl_epochs_run": int(dis["crl"]["epoch"].max()) + 1,
            "down_epochs_run": int(dis["down"]["epoch"].max()) + 1,
        },
        "_meta": {
            "probe_variant": PROBE_VARIANT,
            "vae_run": str(args.vae_run.relative_to(SAVED_CRL)),
            "dis_run": str(args.dis_run.relative_to(SAVED_CRL)),
        },
    }
    (ROOT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
