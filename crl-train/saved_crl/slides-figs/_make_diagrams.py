"""Architecture + training-flow diagrams for the slide deck.

Three figures, one matplotlib script, no graphviz dependency:

  fig6_architecture.png         — forward-pass schematic (multiscale frontend)
  fig7_frontend_multiscale.png  — multiscale frontend internals
  fig9_training_flow.png        — three training modes, shared encoder, mode-specific checkpoints

Style is consistent with _make_figures.py: 300 DPI, 16:9, sans-serif,
viridis-pair accents (multi=#440154), neutral gray for shared parts.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent

C_MULTI = "#440154"
C_SHARED = "#444444"
C_LATENT = "#5b3a87"
C_LOSS = "#b13a3a"
C_BG_MULTI = "#e9defc"
C_BG_SHARED = "#eeeeee"
C_BG_LATENT = "#e7e0f1"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

ARROW_STYLE = "-|>"
ARROW_KW = dict(
    arrowstyle=ARROW_STYLE, mutation_scale=14, color="#222222", linewidth=1.4
)


def box(
    ax,
    xy,
    w,
    h,
    title,
    sub=None,
    *,
    edge=C_SHARED,
    face=C_BG_SHARED,
    title_fs=17,
    sub_fs=14,
    title_weight="bold",
):
    """Draw a labeled rounded rectangle. Returns (cx, cy) center."""
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub is None:
        ax.text(
            cx,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight=title_weight,
            color=edge,
        )
    else:
        ax.text(
            cx,
            cy + 0.10 * h,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight=title_weight,
            color=edge,
        )
        ax.text(
            cx,
            cy - 0.18 * h,
            sub,
            ha="center",
            va="center",
            fontsize=sub_fs,
            color=edge,
            family="monospace",
        )
    return cx, cy


def arrow(ax, p0, p1, *, color="#222222", linewidth=1.4, ls="-", head=True):
    style = ARROW_STYLE if head else "-"
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
        linestyle=ls,
        shrinkA=4,
        shrinkB=4,
    )
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
    box(
        ax,
        (0.2, 3.7),
        2.6,
        1.6,
        "Multi-sensor input",
        sub="audio  (B,1,16000)\nseismic (B,1,200)",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=16,
        sub_fs=13,
    )

    # multiscale frontend
    box(
        ax,
        (3.4, 3.5),
        3.8,
        2.0,
        "Multiscale frontend",
        sub="3 × Conv1D [9, 19, 39]\ncat → 1×1 proj\n→ d_model = 64",
        edge=C_MULTI,
        face=C_BG_MULTI,
        title_fs=17,
        sub_fs=13,
    )

    # token annotation
    ax.text(
        7.85,
        7.2,
        "32 tokens · d_model = 64",
        ha="center",
        va="center",
        fontsize=14,
        color="#666666",
        family="monospace",
    )

    # shared encoder
    box(
        ax,
        (7.8, 3.4),
        3.0,
        2.2,
        "Transformer\nencoder",
        sub="2 layers, 4 heads\nmean-pool → (μ, log σ²)",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=17,
        sub_fs=13,
    )

    # latent posterior
    box(
        ax,
        (11.2, 3.9),
        1.8,
        1.4,
        "z  ~  q(z|x)",
        sub="d_z = 24",
        edge=C_LATENT,
        face=C_BG_LATENT,
        title_fs=17,
        sub_fs=14,
    )

    # latent partitions
    split_x = 13.6
    block_w = 2.4
    block_h = 1.5
    gap = 0.5
    blocks = [
        ("z_signal", "vehicle-relevant"),
        ("z_environment", "nuisance / context"),
    ]
    n = len(blocks)
    total_h = n * block_h + (n - 1) * gap
    y_top = 4.6 + total_h / 2 - block_h
    centers = []
    for i, (name, sub) in enumerate(blocks):
        y = y_top - i * (block_h + gap)
        cx, cy = box(
            ax,
            (split_x, y),
            block_w,
            block_h,
            name,
            sub=sub,
            edge=C_LATENT,
            face=C_BG_LATENT,
            title_fs=17,
            sub_fs=14,
        )
        centers.append((cx, cy))

    # straight-line arrows — input → frontend → encoder → z → splits
    arrow(ax, (2.8, 4.5), (3.4, 4.5), color=C_MULTI, linewidth=1.4)
    arrow(ax, (7.2, 4.5), (7.8, 4.5), color=C_MULTI, linewidth=1.4)
    arrow(ax, (10.8, 4.6), (11.2, 4.6))
    for cx, cy in centers:
        arrow(ax, (13.0, 4.6), (split_x, cy), linewidth=0.9)

    ax.text(
        0.2,
        8.0,
        "Forward pass · CRL pretraining",
        fontsize=15,
        color="#555555",
        style="italic",
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


# ---------- fig 7: multiscale frontend internals ---------------------------


def fig7_multiscale(out: Path) -> None:
    fig, ax = setup_canvas()

    box(
        ax,
        (0.4, 3.7),
        2.2,
        1.6,
        "Per-sensor input",
        sub="(B, C, T)\nfp32",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=16,
        sub_fs=14,
    )

    # three parallel branches
    branch_x = 3.6
    branch_w = 3.6
    branch_h = 1.4
    ks_list = [9, 19, 39]
    rcps = [
        "short kernel · fast events",
        "mid kernel · phoneme-scale",
        "long kernel · vehicle envelope",
    ]
    centers = []
    for i, (k, desc) in enumerate(zip(ks_list, rcps)):
        y = 6.5 - i * 2.2
        cx, cy = box(
            ax,
            (branch_x, y),
            branch_w,
            branch_h,
            f"Conv1D  ks = {k}",
            sub=f"GroupNorm + GELU\n{desc}",
            edge=C_MULTI,
            face=C_BG_MULTI,
            title_fs=17,
            sub_fs=14,
        )
        centers.append((cx, cy))
        arrow(ax, (2.6, 4.5), (branch_x, cy), color=C_MULTI, linewidth=1.4)

    # concat
    cat_x = 8.4
    box(
        ax,
        (cat_x, 3.7),
        2.2,
        1.6,
        "Channel-cat",
        sub="3 · out_C\non channel axis",
        edge=C_MULTI,
        face=C_BG_MULTI,
        title_fs=17,
        sub_fs=14,
    )
    for cx, cy in centers:
        arrow(ax, (branch_x + branch_w, cy), (cat_x, 4.5), color=C_MULTI, linewidth=1.4)

    # 1x1 proj
    box(
        ax,
        (11.2, 3.7),
        2.2,
        1.6,
        "1 × 1 proj",
        sub="→ d_model = 64",
        edge=C_MULTI,
        face=C_BG_MULTI,
        title_fs=17,
        sub_fs=14,
    )
    arrow(ax, (cat_x + 2.2, 4.5), (11.2, 4.5), color=C_MULTI, linewidth=1.4)

    # output
    box(
        ax,
        (14.0, 3.7),
        1.6,
        1.6,
        "to encoder",
        sub="(B, 64, 32)",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=16,
        sub_fs=14,
    )
    arrow(ax, (13.4, 4.5), (14.0, 4.5))

    ax.text(
        0.4,
        8.55,
        "Multiscale frontend · learned multi-receptive-field convs",
        fontsize=15,
        color="#555555",
        style="italic",
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


# ---------- fig 9: training flow + losses + dual checkpoint ---------------


def fig9_training_flow(out: Path) -> None:
    """Three training modes share an encoder; each row carries its own loss
    and checkpoint metric. Strictly horizontal flow, no crossings."""

    # 20.4:10 canvas — wider gutters, row pitch 3.0, rows centered on y = 5.0.
    fig, ax = plt.subplots(figsize=(20.4, 10))
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_axis_off()

    C_VAE = C_LOSS                      # red — VAE/recon family
    C_DIS = C_LATENT                    # purple — disentangled
    C_CON = "#1f7a8c"                   # teal — contrastive
    C_BG_VAE = "#f7e5e5"
    C_BG_DIS = C_BG_LATENT
    C_BG_CON = "#dceef2"

    # Vertical layout — three rows centered on y = 5.0 with equal pitch.
    center_y = 5.0
    row_pitch = 3.0
    row_h = 1.4
    row_y_vae = center_y + row_pitch - row_h / 2          # top row, y of bottom edge
    row_y_dis = center_y - row_h / 2                       # middle row
    row_y_con = center_y - row_pitch - row_h / 2           # bottom row
    row_centers = {
        row_y_vae: row_y_vae + row_h / 2,
        row_y_dis: row_y_dis + row_h / 2,
        row_y_con: row_y_con + row_h / 2,
    }

    # Left column — Inputs / Shared encoder / Latent z, all vertically centered on y = 5.0.
    # Equal horizontal gap (0.6) before AND after the Shared encoder.
    enc_w, enc_h = 2.8, 1.8
    z_w, z_h = 2.0, 1.8
    inputs_right = 1.8
    col_gap = 0.6
    enc_x = inputs_right + col_gap                 # 2.4
    enc_y = center_y - enc_h / 2
    z_x = enc_x + enc_w + col_gap                  # 5.8
    z_y = center_y - z_h / 2

    box(
        ax,
        (0.2, center_y - 0.6),
        1.6,
        1.2,
        "Inputs",
        sub="audio +\nseismic",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=16,
        sub_fs=12,
    )
    box(
        ax,
        (enc_x, enc_y),
        enc_w,
        enc_h,
        "Shared encoder",
        sub="frontend +\ntransformer\n→ μ, log σ²",
        edge=C_SHARED,
        face=C_BG_SHARED,
        title_fs=16,
        sub_fs=12,
    )
    box(
        ax,
        (z_x, z_y),
        z_w,
        z_h,
        "Latent  z",
        sub="pres · type\nprox · env\nfree",
        edge=C_LATENT,
        face=C_BG_LATENT,
        title_fs=16,
        sub_fs=12,
    )
    arrow(ax, (1.8, center_y), (enc_x, center_y))
    arrow(ax, (enc_x + enc_w, center_y), (z_x, center_y))

    z_right = z_x + z_w
    z_mid = (z_right, center_y)

    # Wider horizontal spacing — gap doubled, label gutter added.
    head_w = 2.2
    loss_w = 3.4
    ckpt_w = 3.4
    gap = 0.6

    label_gutter = 1.6                                     # space for VAE/Disent./Contrastive labels
    fan_x = z_right + 0.6
    head_x = fan_x + label_gutter
    loss_x = head_x + head_w + gap
    ckpt_x = loss_x + loss_w + gap

    arrow(ax, z_mid, (fan_x, center_y), color=C_SHARED, linewidth=1.4, head=False)

    for row_y, color in [
        (row_y_vae, C_VAE),
        (row_y_dis, C_DIS),
        (row_y_con, C_CON),
    ]:
        cy = row_centers[row_y]
        arrow(ax, (fan_x, center_y), (fan_x, cy), color=color, linewidth=1.4, head=False)
        arrow(ax, (fan_x, cy), (head_x, cy), color=color, linewidth=1.4)

    def mode_row(y, label, head_title, head_sub, loss_title, loss_sub, ckpt_title, ckpt_sub, edge, face_bg, face_loss):
        cy = y + row_h / 2
        box(
            ax,
            (head_x, y),
            head_w,
            row_h,
            head_title,
            sub=head_sub,
            edge=edge,
            face=face_bg,
            title_fs=14,
            sub_fs=11,
        )
        box(
            ax,
            (loss_x, y),
            loss_w,
            row_h,
            loss_title,
            sub=loss_sub,
            edge=edge,
            face=face_loss,
            title_fs=14,
            sub_fs=11,
        )
        box(
            ax,
            (ckpt_x, y),
            ckpt_w,
            row_h,
            ckpt_title,
            sub=ckpt_sub,
            edge=C_SHARED,
            face=C_BG_SHARED,
            title_fs=13,
            sub_fs=11,
        )
        arrow(ax, (head_x + head_w, cy), (loss_x, cy), color=edge, linewidth=1.4)
        arrow(ax, (loss_x + loss_w, cy), (ckpt_x, cy), color=edge, linewidth=1.4)
        # Mode label sits in the gutter between the fan column and the head box,
        # vertically centered on the row's horizontal arrow — never on the vertical fan line.
        ax.text(
            head_x - 0.15,
            cy + 0.22,
            label,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=edge,
        )

    mode_row(
        row_y_vae,
        "VAE",
        "Decoder +\nAux heads",
        "MSE recon\npres / type",
        "ELBO + Aux",
        "L_recon + β·L_KL\n+ λ_p·L_pres + λ_t·L_type",
        "Dual ckpt",
        "best ← min val_ref_elbo\naux  ← max val_aux_type_f1",
        edge=C_VAE,
        face_bg=C_BG_VAE,
        face_loss="#f7e5e5",
    )

    mode_row(
        row_y_dis,
        "Disentangled",
        "Split latent\n+ paired enc.",
        "z_signal / z_env\n(audio, seismic)",
        "ELBO + Disent.",
        "L_recon + β·L_KL\n+ L_align + L_stable + L_invar",
        "Dual ckpt",
        "best ← min val_ref_elbo\naux  ← max val_aux_type_f1",
        edge=C_DIS,
        face_bg=C_BG_DIS,
        face_loss=C_BG_DIS,
    )

    mode_row(
        row_y_con,
        "Contrastive",
        "Proj. MLP +\nstratified pairs",
        "L2-norm μ\nP_b positives",
        "NT-Xent",
        "L_NTXent\n(temperature τ = 0.1)",
        "Single ckpt",
        "best ← min\nval_contrastive_loss",
        edge=C_CON,
        face_bg=C_BG_CON,
        face_loss=C_BG_CON,
    )

    ax.text(
        0.2,
        9.5,
        "CRL training modes · shared encoder, three objectives, mode-specific checkpoints",
        fontsize=15,
        color="#555555",
        style="italic",
    )
    ax.text(
        0.2,
        9.05,
        "β anneals 0.02 → 1.0 in VAE / Disentangled; val_ref_elbo evaluated at β = 1",
        fontsize=12,
        color="#888888",
        style="italic",
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    fig6_architecture(ROOT / "fig6_architecture.png")
    fig7_multiscale(ROOT / "fig7_frontend_multiscale.png")
    fig9_training_flow(ROOT / "fig9_training_flow.png")
    print("wrote 3 diagram PNGs to", ROOT)


if __name__ == "__main__":
    main()
