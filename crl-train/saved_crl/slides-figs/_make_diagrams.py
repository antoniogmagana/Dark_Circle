"""Architecture + training-flow diagrams for the slide deck.

Three figures, one matplotlib script, no graphviz dependency:

  fig6_architecture.png      — forward-pass schematic, both frontends side by side
  fig7_frontend_multiscale.png — multiscale frontend internals
  fig8_frontend_morlet.png   — morlet (per-sensor) frontend internals
  fig9_training_flow.png     — anchor/partner training graph + losses + dual ckpt

Style is consistent with _make_figures.py: 300 DPI, 16:9, sans-serif,
viridis-pair accents (multi=#440154, morlet=#21918c), neutral gray for shared
parts.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent

C_MULTI = "#440154"
C_MORLET = "#21918c"
C_SHARED = "#444444"
C_LATENT = "#5b3a87"
C_LOSS = "#b13a3a"
C_BG_MULTI = "#e9defc"
C_BG_MORLET = "#d6f0ee"
C_BG_SHARED = "#eeeeee"
C_BG_LATENT = "#e7e0f1"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

ARROW_STYLE = "-|>"
ARROW_KW = dict(arrowstyle=ARROW_STYLE, mutation_scale=14,
                color="#222222", linewidth=1.4)


def box(ax, xy, w, h, title, sub=None, *, edge=C_SHARED, face=C_BG_SHARED,
        title_fs=17, sub_fs=14, title_weight="bold"):
    """Draw a labeled rounded rectangle. Returns (cx, cy) center."""
    x, y = xy
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.05",
                       linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub is None:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight=title_weight, color=edge)
    else:
        ax.text(cx, cy + 0.10 * h, title, ha="center", va="center",
                fontsize=title_fs, fontweight=title_weight, color=edge)
        ax.text(cx, cy - 0.18 * h, sub, ha="center", va="center",
                fontsize=sub_fs, color=edge, family="monospace")
    return cx, cy


def arrow(ax, p0, p1, *, color="#222222", linewidth=1.4, ls="-", head=True):
    style = ARROW_STYLE if head else "-"
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=linewidth, linestyle=ls,
                        shrinkA=4, shrinkB=4)
    ax.add_patch(a)


def setup_canvas(figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


# ---------- fig 6: forward-pass, both frontends ----------------------------

def fig6_architecture(out: Path) -> None:
    fig, ax = setup_canvas()

    # multi-sensor input
    box(ax, (0.2, 3.2), 2.6, 2.6,
        "Multi-sensor input",
        sub="audio  (B,1,16000)\nseismic (B,1,200)",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=16, sub_fs=13)

    # frontend choice — wider boxes to fit the new font sizes
    box(ax, (3.2, 5.5), 4.0, 2.2, "Frontend  (option A)",
        sub="Multiscale\n3 × Conv1D [9,19,39]\ncat → 1×1 proj → d_model",
        edge=C_MULTI, face=C_BG_MULTI, title_fs=17, sub_fs=13)
    box(ax, (3.2, 1.3), 4.0, 2.2, "Frontend  (option B)",
        sub="Morlet (per-sensor)\nwavelet bank, w0=6\nlog_power + cos/sin phase",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=17, sub_fs=13)
    ax.text(5.2, 4.5, "either / or", ha="center", va="center",
            fontsize=14, color="#777777", style="italic")

    # token annotation
    ax.text(7.85, 8.2, "32 tokens · d_model = 64",
            ha="center", va="center", fontsize=14, color="#666666",
            family="monospace")

    # shared encoder
    box(ax, (7.6, 3.4), 3.0, 2.2, "Transformer\nencoder",
        sub="2 layers, 4 heads\nmean-pool → (μ, log σ²)",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=17, sub_fs=13)

    # latent posterior
    box(ax, (11.0, 3.9), 1.8, 1.4, "z  ~  q(z|x)",
        sub="d_z = 24", edge=C_LATENT, face=C_BG_LATENT,
        title_fs=17, sub_fs=14)

    # latent partitions
    split_x = 13.4
    block_w = 2.6
    block_h = 1.5
    gap = 0.5
    blocks = [
        ("z_signal",      "vehicle-relevant"),
        ("z_environment", "nuisance / context"),
    ]
    n = len(blocks)
    total_h = n * block_h + (n - 1) * gap
    y_top = 4.6 + total_h / 2 - block_h
    centers = []
    for i, (name, sub) in enumerate(blocks):
        y = y_top - i * (block_h + gap)
        cx, cy = box(ax, (split_x, y), block_w, block_h, name, sub=sub,
                     edge=C_LATENT, face=C_BG_LATENT, title_fs=17, sub_fs=14)
        centers.append((cx, cy))

    # arrows — input → frontends → encoder → z → splits
    arrow(ax, (2.8, 5.0), (3.2, 6.6), color=C_MULTI, linewidth=1.4)
    arrow(ax, (2.8, 4.2), (3.2, 2.4), color=C_MORLET, linewidth=1.4)
    arrow(ax, (7.2, 6.6), (7.6, 5.0), color=C_MULTI, linewidth=1.4)
    arrow(ax, (7.2, 2.4), (7.6, 4.2), color=C_MORLET, linewidth=1.4)
    arrow(ax, (10.6, 4.6), (11.0, 4.6))
    for cx, cy in centers:
        arrow(ax, (12.8, 4.6), (split_x, cy), linewidth=0.9)

    ax.text(0.2, 8.6, "Forward pass · CRL pretraining (frontend is the swap point)",
            fontsize=15, color="#555555", style="italic")
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


# ---------- fig 7: multiscale frontend internals ---------------------------

def fig7_multiscale(out: Path) -> None:
    fig, ax = setup_canvas()

    box(ax, (0.4, 3.7), 2.2, 1.6, "Per-sensor input",
        sub="(B, C, T)\nfp32", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=16, sub_fs=14)

    # three parallel branches
    branch_x = 3.6
    branch_w = 3.6
    branch_h = 1.4
    ks_list = [9, 19, 39]
    rcps = ["short kernel · fast events",
            "mid kernel · phoneme-scale",
            "long kernel · vehicle envelope"]
    centers = []
    for i, (k, desc) in enumerate(zip(ks_list, rcps)):
        y = 6.5 - i * 2.2
        cx, cy = box(ax, (branch_x, y), branch_w, branch_h,
                     f"Conv1D  ks = {k}",
                     sub=f"GroupNorm + GELU\n{desc}",
                     edge=C_MULTI, face=C_BG_MULTI,
                     title_fs=17, sub_fs=14)
        centers.append((cx, cy))
        arrow(ax, (2.6, 4.5), (branch_x, cy), color=C_MULTI, linewidth=1.4)

    # concat
    cat_x = 8.4
    box(ax, (cat_x, 3.7), 2.2, 1.6, "Channel-cat",
        sub="3 · out_C\non channel axis",
        edge=C_MULTI, face=C_BG_MULTI, title_fs=17, sub_fs=14)
    for cx, cy in centers:
        arrow(ax, (branch_x + branch_w, cy), (cat_x, 4.5),
              color=C_MULTI, linewidth=1.4)

    # 1x1 proj
    box(ax, (11.2, 3.7), 2.2, 1.6, "1 × 1 proj",
        sub="→ d_model = 64",
        edge=C_MULTI, face=C_BG_MULTI, title_fs=17, sub_fs=14)
    arrow(ax, (cat_x + 2.2, 4.5), (11.2, 4.5), color=C_MULTI, linewidth=1.4)

    # output
    box(ax, (14.0, 3.7), 1.6, 1.6, "to encoder",
        sub="(B, 64, 32)", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=16, sub_fs=14)
    arrow(ax, (13.4, 4.5), (14.0, 4.5))

    ax.text(0.4, 8.55, "Multiscale frontend · learned multi-receptive-field convs",
            fontsize=15, color="#555555", style="italic")
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


# ---------- fig 8: morlet frontend internals -------------------------------

def fig8_morlet(out: Path) -> None:
    fig, ax = setup_canvas()

    box(ax, (0.4, 3.7), 2.2, 1.6, "Per-sensor input",
        sub="(B, 1, T)\naudio T = 16k\nseismic T = 200",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=16, sub_fs=14)

    # filter bank
    box(ax, (3.4, 3.5), 4.0, 2.0, "Morlet filter bank",
        sub="log-spaced freqs · w0 = 6\naudio   ks = 4585  stride 500\nseismic ks = 573   stride 6",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=17, sub_fs=14)
    arrow(ax, (2.6, 4.5), (3.4, 4.5), color=C_MORLET, linewidth=1.4)

    # complex output → three channels
    box(ax, (8.4, 6.0), 2.8, 1.4, "log_power",
        sub="|Wx|² → log",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=17, sub_fs=14)
    box(ax, (8.4, 3.8), 2.8, 1.4, "cos_phase",
        sub="re / |Wx|",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=17, sub_fs=14)
    box(ax, (8.4, 1.6), 2.8, 1.4, "sin_phase",
        sub="im / |Wx|",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=17, sub_fs=14)

    arrow(ax, (7.4, 5.0), (8.4, 6.7), color=C_MORLET, linewidth=1.4)
    arrow(ax, (7.4, 4.5), (8.4, 4.5), color=C_MORLET, linewidth=1.4)
    arrow(ax, (7.4, 4.0), (8.4, 2.3), color=C_MORLET, linewidth=1.4)

    # concat / pool to tokens
    box(ax, (11.8, 3.7), 2.2, 1.6, "channel-cat\n+ pool",
        sub="3 × out_C\n→ 32 tokens",
        edge=C_MORLET, face=C_BG_MORLET, title_fs=16, sub_fs=13)
    arrow(ax, (11.2, 6.7), (11.8, 4.8), color=C_MORLET, linewidth=1.2)
    arrow(ax, (11.2, 4.5), (11.8, 4.5), color=C_MORLET, linewidth=1.2)
    arrow(ax, (11.2, 2.3), (11.8, 4.2), color=C_MORLET, linewidth=1.2)

    box(ax, (14.2, 3.7), 1.6, 1.6, "to encoder",
        sub="(B, C, 32)", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=15, sub_fs=13)
    arrow(ax, (14.0, 4.5), (14.2, 4.5))

    ax.text(0.4, 8.55,
            "Morlet (per-sensor) frontend · fixed wavelet bank with phase channel",
            fontsize=15, color="#555555", style="italic")
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


# ---------- fig 9: training flow + losses + dual checkpoint ---------------

def fig9_training_flow(out: Path) -> None:
    fig, ax = setup_canvas(figsize=(16, 9))

    # ---- column 1: inputs ----
    box(ax, (0.2, 6.6), 2.0, 1.3, "Anchor x_t",
        sub="audio + seismic", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=16, sub_fs=13)
    box(ax, (0.2, 1.1), 2.0, 1.3, "Partner x_tn",
        sub="stratified pair", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=16, sub_fs=13)

    # ---- column 2: shared encoder applied twice ----
    box(ax, (2.6, 6.4), 2.6, 1.7, "Encoder",
        sub="frontend +\ntransformer → z_t",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=17, sub_fs=13)
    box(ax, (2.6, 0.9), 2.6, 1.7, "Encoder (shared)",
        sub="→ z_tn", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=17, sub_fs=13)
    arrow(ax, (2.2, 7.25), (2.6, 7.25))
    arrow(ax, (2.2, 1.75), (2.6, 1.75))

    # ---- column 3: latent splits ----
    box(ax, (5.6, 6.4), 2.4, 1.7, "z_t  split",
        sub="pres · type\nprox · env · free",
        edge=C_LATENT, face=C_BG_LATENT, title_fs=17, sub_fs=13)
    box(ax, (5.6, 0.9), 2.4, 1.7, "z_tn  split",
        sub="env block used",
        edge=C_LATENT, face=C_BG_LATENT, title_fs=17, sub_fs=13)
    arrow(ax, (5.2, 7.25), (5.6, 7.25))
    arrow(ax, (5.2, 1.75), (5.6, 1.75))

    # right-edge waypoints on the split boxes
    z_t_right_top = (8.0, 7.7)
    z_t_right_mid = (8.0, 7.3)
    z_t_right_bot = (8.0, 6.9)
    z_tn_right    = (8.0, 1.75)

    # ---- column 4: downstream consumers ----
    dec_x, dec_y = 9.0, 7.3
    box(ax, (dec_x, dec_y), 2.4, 1.2, "Decoder",
        sub="MSE recon → x̂",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=16, sub_fs=13)
    arrow(ax, z_t_right_top, (dec_x, dec_y + 0.6))

    aux_x, aux_y = 9.0, 5.5
    box(ax, (aux_x, aux_y), 2.4, 1.2, "Aux heads",
        sub="pres · type · prox", edge=C_SHARED, face=C_BG_SHARED,
        title_fs=16, sub_fs=13)
    arrow(ax, z_t_right_bot, (aux_x, aux_y + 0.6))

    interv_x, interv_y = 9.0, 3.0
    interv_w, interv_h = 2.4, 1.6
    box(ax, (interv_x, interv_y), interv_w, interv_h,
        "Intervention\nclassifier",
        sub="MLP on\n[z_env_t, z_env_tn]",
        edge=C_LATENT, face=C_BG_LATENT, title_fs=16, sub_fs=13)

    # waypoint x: between split boxes and consumer column, away from the box
    way_x = 8.55
    # anchor env feed
    arrow(ax, z_t_right_mid, (way_x, 7.3), color=C_LATENT, linewidth=1.4, head=False)
    arrow(ax, (way_x, 7.3), (way_x, interv_y + interv_h - 0.4),
          color=C_LATENT, linewidth=1.4, head=False)
    arrow(ax, (way_x, interv_y + interv_h - 0.4),
          (interv_x, interv_y + interv_h - 0.4),
          color=C_LATENT, linewidth=1.4)
    # partner env feed
    arrow(ax, z_tn_right, (way_x, 1.75), color=C_LATENT, linewidth=1.4, head=False)
    arrow(ax, (way_x, 1.75), (way_x, interv_y + 0.4),
          color=C_LATENT, linewidth=1.4, head=False)
    arrow(ax, (way_x, interv_y + 0.4), (interv_x, interv_y + 0.4),
          color=C_LATENT, linewidth=1.4)

    # ---- column 5: loss block ----
    loss_x, loss_y, loss_w, loss_h = 12.0, 4.0, 3.6, 2.6
    box(ax, (loss_x, loss_y), loss_w, loss_h,
        "ELBO + Aux + Interv",
        sub="recon · β · KL\n+ pres / type / prox\n+ BCE(intervention)",
        edge=C_LOSS, face="#f7e5e5", title_fs=17, sub_fs=13)
    arrow(ax, (dec_x + 2.4, dec_y + 0.6), (loss_x, loss_y + loss_h - 0.5),
          color=C_LOSS, linewidth=1.4)
    arrow(ax, (aux_x + 2.4, aux_y + 0.6), (loss_x, loss_y + loss_h / 2),
          color=C_LOSS, linewidth=1.4)
    arrow(ax, (interv_x + interv_w, interv_y + interv_h - 0.4),
          (loss_x, loss_y + 0.5), color=C_LOSS, linewidth=1.4)

    # ---- column 6: dual checkpoint ----
    ckpt_x, ckpt_y, ckpt_w, ckpt_h = 12.0, 1.0, 3.6, 1.8
    box(ax, (ckpt_x, ckpt_y), ckpt_w, ckpt_h,
        "Dual checkpoint",
        sub="crl_best.pth      ← min val_ref_elbo\ncrl_best_aux_type ← max val_aux_type_f1",
        edge=C_SHARED, face=C_BG_SHARED, title_fs=16, sub_fs=12)
    arrow(ax, (loss_x + loss_w / 2, loss_y),
          (ckpt_x + ckpt_w / 2, ckpt_y + ckpt_h),
          color=C_LOSS, linewidth=1.4)

    ax.text(0.2, 8.6,
            "CRL training graph · anchor + partner + dual checkpoint  ·  "
            "β anneals 0.02 → 1.0; val_ref_elbo evaluated at β = 1",
            fontsize=14, color="#555555", style="italic")
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    fig6_architecture(ROOT / "fig6_architecture.png")
    fig7_multiscale(ROOT / "fig7_frontend_multiscale.png")
    fig8_morlet(ROOT / "fig8_frontend_morlet.png")
    fig9_training_flow(ROOT / "fig9_training_flow.png")
    print("wrote 4 diagram PNGs to", ROOT)


if __name__ == "__main__":
    main()
