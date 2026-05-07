"""Render the dual-bundle CRL inference data-flow diagram.

Boxes:
  ROS2 sensor topics -> ingestor -> NATS sensor.data
                                          |
                +-------------------------+-------------------------+
                |                                                   |
   detect bundle (multiscale-vae)                 classify bundle (multiscale-disentangled)
   frontend -> encoder -> presence head           frontend -> encoder -> linear_signal type head
                |                                                   |
       NATS detection.result -------(if present)----------> [re-encode from waveform]
                                                                    |
                                                          NATS classification.result
                                                                    |
                                                                  egress -> /inference_result
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def box(ax, x, y, w, h, text, *, fc="#dde8f3", ec="#1f3a5f", fontsize=9, lw=1.3):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, *, label=None, label_offset=(0, 0.05), color="#333", lw=1.4):
    ax.annotate(
        "",
        xy=(x2, y2), xycoords="data",
        xytext=(x1, y1), textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
    )
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label, ha="center", va="center", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )


def main() -> int:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color palette: shared infra muted, detect-side blue, classify-side red.
    INFRA = "#e9ecef"
    DETECT = "#cfe2ff"
    CLASSIFY = "#f8d7da"
    ENCODER = "#fff3cd"

    # Top: sensor input flow (centered)
    box(ax, 3.5, 7.1, 3.0, 0.55, "ROS2 sensor topics\n(audio 16 kHz · seismic 100 Hz)", fc=INFRA)
    box(ax, 3.5, 6.2, 3.0, 0.55, "ingestor pod\n1-sec window · DC removal", fc=INFRA)
    box(ax, 3.5, 5.3, 3.0, 0.45, "NATS  sensor.data", fc=INFRA, fontsize=9)

    arrow(ax, 5.0, 7.1, 5.0, 6.78)
    arrow(ax, 5.0, 6.2, 5.0, 5.78)

    # Two columns: detect side (left), classify side (right).
    detect_x, classify_x = 0.4, 6.6
    col_w = 3.0

    # Detect column.
    box(ax, detect_x, 4.3, col_w, 0.55, "infer-detect pod", fc=DETECT, fontsize=10)
    box(ax, detect_x + 0.15, 3.45, col_w - 0.3, 0.55,
        "encoder_fused.ts\n(VAE, d_z=32)", fc=ENCODER, fontsize=8)
    box(ax, detect_x + 0.15, 2.6, col_w - 0.3, 0.55,
        "presence head\nLinear(D_PRES=4 → 1)", fc=ENCODER, fontsize=8)
    box(ax, detect_x, 1.7, col_w, 0.45, "NATS  detection.result", fc=INFRA, fontsize=9)

    arrow(ax, detect_x + col_w / 2, 4.3, detect_x + col_w / 2, 4.0)
    arrow(ax, detect_x + col_w / 2, 3.45, detect_x + col_w / 2, 3.15)
    arrow(ax, detect_x + col_w / 2, 2.6, detect_x + col_w / 2, 2.15)

    # NATS sensor.data branches into both pods.
    arrow(ax, 3.5 + 1.5, 5.3, detect_x + col_w / 2, 4.85)
    arrow(ax, 3.5 + 1.5, 5.3, classify_x + col_w / 2, 4.85)

    # Classify column. Note the re-encode design: same waveform passes through
    # a separate encoder.
    box(ax, classify_x, 4.3, col_w, 0.55, "infer-classify pod", fc=CLASSIFY, fontsize=10)
    box(ax, classify_x + 0.15, 3.45, col_w - 0.3, 0.55,
        "encoder_fused.ts\n(disentangled, d_z=24, d_signal=12)", fc=ENCODER, fontsize=8)
    box(ax, classify_x + 0.15, 2.6, col_w - 0.3, 0.55,
        "type_head_fused.ts\nLinear(z[0:12] → 4 classes)", fc=ENCODER, fontsize=8)
    box(ax, classify_x, 1.7, col_w, 0.45, "NATS  classification.result", fc=INFRA, fontsize=9)

    arrow(ax, classify_x + col_w / 2, 4.3, classify_x + col_w / 2, 4.0)
    arrow(ax, classify_x + col_w / 2, 3.45, classify_x + col_w / 2, 3.15)
    arrow(ax, classify_x + col_w / 2, 2.6, classify_x + col_w / 2, 2.15)

    # Detection.result -> classify pod gate (only positives flow further).
    arrow(
        ax, detect_x + col_w, 1.92, classify_x, 1.92,
        label="if vehicle_detected", label_offset=(0, 0.18),
    )

    # Egress at the bottom.
    box(ax, 3.5, 0.6, 3.0, 0.55, "egress pod\nROS2  /inference_result", fc=INFRA, fontsize=9)
    arrow(ax, detect_x + col_w / 2, 1.7, 4.5, 1.15)
    arrow(ax, classify_x + col_w / 2, 1.7, 5.5, 1.15)

    # Title
    ax.text(5.0, 7.85, "Dual-bundle CRL inference data flow",
            ha="center", va="center", fontsize=12, fontweight="bold")

    out = Path("/Users/brandontaylor/Documents/CMU/CAPSTONE/Dark_Circle/crl-train/saved_crl/analysis/dual_bundle_dataflow")
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    print(f"wrote {out}.png and {out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
