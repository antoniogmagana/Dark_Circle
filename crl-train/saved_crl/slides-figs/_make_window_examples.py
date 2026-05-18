"""Render one real (audio, seismic) window pair with multiscale kernels overlaid.

Output: fig4_window_examples.png — two stacked panels (audio at 16 kHz, seismic
at 100 Hz) sharing the same 1-second window. Highlighted spans on the audio
panel show the receptive field of each multiscale Conv1D kernel size; a
matching span on the seismic panel shows the same receptive-field idea at the
seismic rate. The kernel sizes are exaggerated (longer than the on-disk
defaults) so they read clearly in print.

Reads waveform tensors directly from the on-disk cache that
`SensorDataset(..., cache_dir=...)` writes:

    saved_crl/caches/waveform/{dataset}_{sensor}_{vehicle}_{rs}_sr{rate}.pt

Each .pt is `{amplitude: (n_windows, samples_per_window), present: (n_windows,)}`.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SAVED_CRL = ROOT.parent
CACHE = SAVED_CRL / "caches" / "waveform"

AUDIO_SR = 16000
SEISMIC_SR = 100

# One paired stem chosen for clarity: IOBT silverado / rs1 — Chevrolet
# Silverado pass with strong, transient audio and visible ground motion. The
# specific window index is pre-selected (rather than first-present) because
# we want a clean, presentation-quality example, not a representative one.
DATASET = "iobt"
VEHICLE = "silverado0315pm"
RS_NODE = "rs1"
WINDOW_IDX = 604
DATASET_LABEL = "IOBT"
VEHICLE_LABEL = "Silverado pass"

# Exaggerated kernel sizes (samples) to make receptive fields visible at print
# scale. Real defaults are [9, 19, 39, 159] on audio; we scale them up by ~50×
# so the spans are readable on a 1-second window. The 1×1 ratio between audio
# and seismic kernels matches the cfg's "1 sample at audio rate ≈ same time
# fraction as 1 sample at seismic rate" idea.
AUDIO_KERNELS = [400, 1600, 4800]   # ~25 ms · 100 ms · 300 ms at 16 kHz
SEISMIC_KERNELS = [3, 10, 30]        # ~30 ms · 100 ms · 300 ms at 100 Hz

KERNEL_LABELS = [
    "short kernel",
    "mid kernel",
    "long kernel",
]

# Highlight colors for the three kernel sizes (sequential viridis sample).
KERNEL_COLORS = ["#fde725", "#5ec962", "#3b528b"]
KERNEL_EDGES = ["#bfa800", "#2c7a3a", "#1f2f5e"]

C_AUDIO = "#440154"     # viridis dark-purple
C_SEISMIC = "#21918c"   # viridis teal-green

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_pair() -> tuple[np.ndarray, np.ndarray, int]:
    """Load the pre-selected (audio_window, seismic_window, window_index) pair,
    de-meaned per channel as the model pipeline does before the frontend."""
    audio_path = CACHE / f"{DATASET}_audio_{VEHICLE}_{RS_NODE}_sr{AUDIO_SR}.pt"
    seismic_path = CACHE / f"{DATASET}_seismic_{VEHICLE}_{RS_NODE}_sr{SEISMIC_SR}.pt"
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    a = torch.load(audio_path, weights_only=False, map_location="cpu")
    s = torch.load(seismic_path, weights_only=False, map_location="cpu")
    pres = bool(a["present"][WINDOW_IDX].item()) and bool(s["present"][WINDOW_IDX].item())
    if not pres:
        raise ValueError(
            f"window {WINDOW_IDX} in {DATASET}/{VEHICLE}/{RS_NODE} is not flagged present"
        )
    audio = a["amplitude"][WINDOW_IDX].numpy()
    seismic = s["amplitude"][WINDOW_IDX].numpy()
    # Per-window mean subtraction — matches `transforms.remove_dc` applied
    # once on raw windows before the frontend. This is what the model sees.
    audio = audio - audio.mean()
    seismic = seismic - seismic.mean()
    return audio, seismic, WINDOW_IDX


def overlay_kernel_windows(
    ax,
    sample_rate: int,
    kernel_sizes: list[int],
    y_lim: tuple[float, float],
) -> None:
    """Draw full-height kernel-window boxes on the waveform axis itself.

    Each kernel size gets one rectangle that spans the kernel's receptive-field
    width horizontally and the entire amplitude range vertically (top to bottom
    of the panel). The kernel label sits centered at the top edge of each box.
    """
    anchors = [0.20, 0.50, 0.80]
    y0, y1 = y_lim
    span = y1 - y0
    label_y = y1 + span * 0.04   # just above the top spine
    for ks, anchor, color, edge, label in zip(
        kernel_sizes, anchors, KERNEL_COLORS, KERNEL_EDGES, KERNEL_LABELS
    ):
        width_s = ks / sample_rate
        x0 = max(0.0, anchor - width_s / 2)
        x1 = min(1.0, anchor + width_s / 2)
        ax.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, span,
            facecolor=color, edgecolor=edge, linewidth=1.2, alpha=0.22,
            zorder=3,
        ))
        # Vertical edges drawn solid so the box reads as a window even when
        # the fill is faint over the waveform.
        ax.plot([x0, x0], [y0, y1], color=edge, linewidth=1.2, zorder=4)
        ax.plot([x1, x1], [y0, y1], color=edge, linewidth=1.2, zorder=4)
        ax.text(
            (x0 + x1) / 2, label_y,
            f"{label}  (ks={ks})",
            ha="center", va="bottom",
            fontsize=10, color=edge, fontweight="bold",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor=edge, linewidth=0.8, pad=2.0),
        )


def main() -> None:
    audio, seismic, idx = load_pair()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13.5, 7.5),
        gridspec_kw=dict(hspace=0.55),
    )
    ax_a, ax_s = axes

    t_audio = np.linspace(0.0, 1.0, AUDIO_SR, endpoint=False)
    t_seismic = np.linspace(0.0, 1.0, SEISMIC_SR, endpoint=False)

    audio_max = float(np.max(np.abs(audio)))
    seismic_max = float(np.max(np.abs(seismic)))
    audio_ylim = (-1.05 * audio_max, 1.05 * audio_max)
    seismic_ylim = (-1.05 * seismic_max, 1.05 * seismic_max)

    # ---- Audio panel ----
    ax_a.plot(t_audio, audio, color=C_AUDIO, linewidth=0.7, zorder=2)
    ax_a.set_ylim(audio_ylim)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.axhline(0.0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax_a.set_ylabel("audio amplitude\n(de-meaned)", fontsize=12)
    ax_a.set_title(
        f"Audio · {AUDIO_SR // 1000} kHz · multiscale receptive fields overlaid",
        fontsize=14, pad=22,   # extra pad so kernel labels do not collide
    )
    ax_a.yaxis.grid(True, alpha=0.2)
    ax_a.set_axisbelow(True)
    overlay_kernel_windows(ax_a, AUDIO_SR, AUDIO_KERNELS, audio_ylim)

    # ---- Seismic panel ----
    ax_s.plot(t_seismic, seismic, color=C_SEISMIC, linewidth=1.2, marker="o", markersize=2.5, zorder=2)
    ax_s.set_ylim(seismic_ylim)
    ax_s.set_xlim(0.0, 1.0)
    ax_s.axhline(0.0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax_s.set_ylabel("seismic amplitude\n(de-meaned)", fontsize=12)
    ax_s.set_xlabel("time within window (s)")
    ax_s.set_title(
        f"Seismic · {SEISMIC_SR} Hz · matching receptive fields overlaid",
        fontsize=14, pad=22,
    )
    ax_s.yaxis.grid(True, alpha=0.2)
    ax_s.set_axisbelow(True)
    overlay_kernel_windows(ax_s, SEISMIC_SR, SEISMIC_KERNELS, seismic_ylim)

    # Run-identity annotation in the lower right.
    fig.text(
        0.99, 0.005,
        f"{DATASET} · {VEHICLE} · {RS_NODE} · window {idx}",
        ha="right", va="bottom",
        fontsize=9, color="#888888", family="monospace",
    )

    fig.suptitle(
        f"Paired (audio, seismic) window — {DATASET_LABEL} {VEHICLE_LABEL} · "
        "kernel sizes exaggerated for visibility",
        fontsize=14, color="#555555", style="italic", y=0.995,
    )

    out = ROOT / "fig4_window_examples.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
