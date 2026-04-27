"""Anatomy of the multiscale frontend — three-panel slide diagram.

Shows how the three parallel Conv1D branches in
crl_vehicle/models/frontend.py::MultiScale1DFrontend
(kernel_sizes=[9, 19, 39], stride=4) each sample a different time-scale of
the same input.

Top row of each panel: a shared synthetic input waveform — a fast transient
followed by a chirp and a slow envelope, designed to contain structure on
multiple time-scales.

Bottom of each panel: a single receptive-field window of the corresponding
kernel size highlighted on the waveform, plus the local mean amplitude that
a single Conv1D output sees inside that window. Short kernel → captures
fast events only. Long kernel → captures the slow envelope.

Output: fig11_multiscale_anatomy.png — 16:9, 300 DPI, presentation fonts.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

C_INPUT = "#444444"
# Window fills (light) and matching darker text/edge tones for readability.
C_KS_FILL = ["#5ec962", "#21918c", "#440154"]   # green → teal → purple
C_KS_FACE = ["#5ec96233", "#21918c33", "#44015433"]
C_KS_TEXT = ["#1f7a3c", "#0f5a55", "#2a0a3a"]   # darker variants for text/lines

KERNEL_SIZES = [9, 19, 39]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def synthetic_signal(n: int = 200) -> np.ndarray:
    """Multi-scale toy signal: short transient + mid chirp + long envelope."""
    t = np.arange(n)
    transient = np.exp(-((t - 30) ** 2) / (2 * 3.0 ** 2)) * 0.7
    chirp_freq = 0.05 + 0.10 * (t / n)
    chirp = np.sin(2 * np.pi * chirp_freq * t) * np.exp(-((t - 110) ** 2) / (2 * 18.0 ** 2)) * 0.8
    envelope = np.exp(-((t - 160) ** 2) / (2 * 28.0 ** 2)) * 0.9
    sig = transient + chirp + envelope
    sig = sig / np.max(np.abs(sig))
    return sig


def main() -> None:
    n = 200
    sig = synthetic_signal(n)
    x = np.arange(n)

    # Pick a window center that intersects all the structure (~ chirp peak),
    # so the three kernel widths visibly cover different amounts of context.
    center = 110

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0))

    for i, (ks, fill, face, text_color) in enumerate(
            zip(KERNEL_SIZES, C_KS_FILL, C_KS_FACE, C_KS_TEXT)):
        ax = axes[i]

        # Plot the shared input signal
        ax.plot(x, sig, color=C_INPUT, linewidth=1.8, zorder=2)
        ax.fill_between(x, sig, 0, color=C_INPUT, alpha=0.06, zorder=1)

        # Highlight the receptive-field window centered at the same point.
        # Fill stays light; outlines/lines use the darker variant for contrast.
        half = ks // 2
        x0, x1 = center - half, center + half
        ax.axvspan(x0, x1, color=fill, alpha=0.22, zorder=0)
        ax.axvline(x0, color=text_color, linewidth=1.6, alpha=0.85, zorder=1)
        ax.axvline(x1, color=text_color, linewidth=1.6, alpha=0.85, zorder=1)

        # Width annotation above the window inside the plot
        ax.annotate("", xy=(x1, 1.30), xytext=(x0, 1.30),
                    arrowprops=dict(arrowstyle="<->", color=text_color,
                                    linewidth=1.6))
        ax.text((x0 + x1) / 2, 1.40, f"{ks} samples",
                ha="center", va="bottom", fontsize=13, color=text_color,
                fontweight="bold")

        ax.set_title(f"kernel size = {ks}",
                     fontweight="bold", color=text_color, pad=12)
        ax.set_xlabel("sample index")
        if i == 0:
            ax.set_ylabel("amplitude")
        ax.set_xlim(0, n - 1)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)

        # Descriptor (bottom-left, away from the width arrow at the top)
        descriptors = {
            9:  "short kernel\ncatches fast transients only",
            19: "mid kernel\na few cycles of local oscillation",
            39: "long kernel\nsees the slow envelope",
        }
        ax.text(0.02, 0.04, descriptors[ks],
                transform=ax.transAxes, fontsize=14, color=text_color,
                va="bottom", ha="left", fontweight="bold")

    # Footer
    fig.text(0.5, -0.02,
             "shared input  ·  each Conv1D output sees only the colored window  ·  "
             "stride = 4",
             ha="center", va="top", fontsize=14, color="#555555",
             family="monospace")

    fig.tight_layout(pad=1.5)
    out = ROOT / "fig11_multiscale_anatomy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
