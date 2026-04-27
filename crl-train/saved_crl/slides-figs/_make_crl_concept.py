"""Causal Representation Learning — concept diagram.

Two-column figure:

  Left:  an "entangled" latent — every dimension mixed with every factor of
         variation. Rendered as a colorful blob with crisscrossing connections.

  Right: a CRL-style "factorized" latent — the same dimensions but partitioned
         into named, semantically meaningful blocks (z_signal, z_environment).
         Each block has a single job.

The point: CRL = put structure into the latent space so that downstream
factors are read from dedicated subspaces, not entangled across all dims.

Output: fig12_crl_concept.png — 16:9, 300 DPI, presentation fonts.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent

C_INPUT       = "#444444"
C_TANGLED     = "#9c5fb1"   # mid purple — entangled blob
C_TANGLED_BG  = "#f0e6f4"
C_SIGNAL      = "#1f7a3c"   # darker green
C_SIGNAL_BG   = "#dff2e3"
C_ENV         = "#0f5a55"   # darker teal
C_ENV_BG      = "#daeae9"
C_FACTOR      = "#444444"
C_DECISION    = "#b13a3a"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def rounded_box(ax, xy, w, h, *, edge, face, lw=1.4):
    p = FancyBboxPatch(xy, w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(p)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ---------- shared input column (far left) -----------------------------
    rounded_box(ax, (0.2, 3.4), 2.0, 2.2, edge=C_INPUT, face="#eeeeee")
    ax.text(1.2, 4.85, "x  (input)",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_INPUT)

    # The factors of variation that the input contains — listed inside the
    # input box rather than as floating labels.
    ax.text(1.2, 4.05, "vehicle type\nproximity\nenvironment",
            ha="center", va="center", fontsize=11, color=C_FACTOR,
            style="italic", linespacing=1.3)

    # ---------- LEFT side: entangled / vanilla representation --------------
    # title strip
    ax.text(4.6, 8.4, "vanilla representation",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=C_TANGLED)
    ax.text(4.6, 7.85, "every factor mixed across every latent dim",
            ha="center", va="center", fontsize=14, color="#555555",
            style="italic")

    # tangled blob — single big rounded box with random "wires" inside
    rounded_box(ax, (2.8, 2.0), 3.6, 5.0, edge=C_TANGLED, face=C_TANGLED_BG, lw=2.0)
    ax.text(4.6, 2.45, "z  (entangled)", ha="center", va="center",
            fontsize=18, fontweight="bold", color=C_TANGLED)

    # crisscrossing wires inside the blob to evoke "tangled"
    rng = np.random.default_rng(7)
    n_wires = 24
    for _ in range(n_wires):
        x_pts = rng.uniform(3.0, 6.2, size=4)
        y_pts = rng.uniform(3.1, 6.7, size=4)
        ax.plot(x_pts, y_pts, color=C_TANGLED, alpha=0.35, linewidth=1.2)
    # nodes
    for _ in range(14):
        cx = rng.uniform(3.05, 6.15)
        cy = rng.uniform(3.05, 6.85)
        ax.add_patch(Circle((cx, cy), 0.10, color=C_TANGLED, alpha=0.85))

    # single short arrow from input → tangled blob
    a = FancyArrowPatch((2.2, 4.5), (2.8, 4.5),
                        arrowstyle="-|>", mutation_scale=18,
                        color=C_TANGLED, alpha=0.85, linewidth=1.6,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

    # downstream readout from tangled
    rounded_box(ax, (3.0, 0.6), 3.2, 1.0, edge=C_DECISION, face="#f7e5e5")
    ax.text(4.6, 1.1, "linear readout",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DECISION)
    a = FancyArrowPatch((4.6, 2.0), (4.6, 1.6),
                        arrowstyle="-|>", mutation_scale=16,
                        color=C_DECISION, linewidth=1.4)
    ax.add_patch(a)
    ax.text(4.6, 0.20, "must untangle every factor at once",
            ha="center", va="center", fontsize=13, color=C_DECISION,
            style="italic")

    # ---------- vertical divider -------------------------------------------
    ax.plot([8.0, 8.0], [1.5, 8.6], color="#cccccc", linewidth=1.2,
            linestyle=":")

    # ---------- RIGHT side: causal / factorized representation -------------
    ax.text(11.8, 8.4, "causal representation",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=C_SIGNAL)
    ax.text(11.8, 7.85, "the latent is partitioned by what each block describes",
            ha="center", va="center", fontsize=14, color="#555555",
            style="italic")

    # outer z box
    rounded_box(ax, (8.6, 2.0), 6.4, 5.0, edge=C_SIGNAL, face="#f5fbf6", lw=2.0)
    ax.text(11.8, 2.45, "z  (factorized)", ha="center", va="center",
            fontsize=18, fontweight="bold", color=C_SIGNAL)

    # two inner blocks: signal and environment
    rounded_box(ax, (9.0, 4.6), 2.8, 2.0,
                edge=C_SIGNAL, face=C_SIGNAL_BG, lw=1.6)
    ax.text(10.4, 5.95, "z_signal",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_SIGNAL)
    ax.text(10.4, 5.30, "vehicle-relevant\n(type · proximity)",
            ha="center", va="center", fontsize=13, color=C_SIGNAL)

    rounded_box(ax, (11.8, 4.6), 3.0, 2.0,
                edge=C_ENV, face=C_ENV_BG, lw=1.6)
    ax.text(13.3, 5.95, "z_environment",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_ENV)
    ax.text(13.3, 5.30, "nuisance / context\n(scene, weather, time)",
            ha="center", va="center", fontsize=13, color=C_ENV)

    # Route input → factorized z over the top of the figure so the line
    # doesn't pass through the entangled blob. Three legs: up from input,
    # across above both columns, down into the right z-box.
    over_y = 7.55
    a1 = FancyArrowPatch((1.2, 5.6), (1.2, over_y),
                         arrowstyle="-", color=C_SIGNAL, alpha=0.65,
                         linewidth=1.6)
    ax.add_patch(a1)
    a2 = FancyArrowPatch((1.2, over_y), (11.8, over_y),
                         arrowstyle="-", color=C_SIGNAL, alpha=0.65,
                         linewidth=1.6)
    ax.add_patch(a2)
    a3 = FancyArrowPatch((11.8, over_y), (11.8, 7.0),
                         arrowstyle="-|>", mutation_scale=18,
                         color=C_SIGNAL, alpha=0.85, linewidth=1.6)
    ax.add_patch(a3)

    # (block sub-text already names what each block captures; no extra labels)

    # downstream readout from z_signal only
    rounded_box(ax, (8.9, 0.6), 3.0, 1.0, edge=C_DECISION, face="#f7e5e5")
    ax.text(10.4, 1.1, "linear readout",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DECISION)
    a = FancyArrowPatch((10.4, 4.6), (10.4, 1.6),
                        arrowstyle="-|>", mutation_scale=16,
                        color=C_DECISION, linewidth=1.4)
    ax.add_patch(a)
    ax.text(10.4, 0.20, "reads only z_signal",
            ha="center", va="center", fontsize=13, color=C_DECISION,
            style="italic")

    # "ignored" label coming off z_environment
    a = FancyArrowPatch((13.3, 4.6), (13.3, 2.0),
                        arrowstyle="-", color="#999999", linewidth=1.0,
                        linestyle=(0, (3, 3)))
    ax.add_patch(a)
    ax.text(13.3, 1.1, "free to vary,\nnot used at decision time",
            ha="center", va="center", fontsize=12, color="#777777",
            style="italic")

    # title bar
    ax.text(0.2, 8.6,
            "Causal Representation Learning · structure the latent so each "
            "factor lives in its own subspace",
            fontsize=15, color="#555555", style="italic")

    fig.tight_layout(pad=0.5)
    out = ROOT / "fig12_crl_concept.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
