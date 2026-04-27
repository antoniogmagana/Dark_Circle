"""Anatomy of a Morlet wavelet — three-panel slide diagram.

Curves computed from the same formula used in
crl_vehicle/models/frontend.py::MorletFilterbank._build_kernels:

    s     = w0 / (2π · f)                # scale (seconds)
    norm  = (π · s)^(-1/4)               # admissibility normalization
    gauss = norm · exp(-½ · (t/s)²)      # Gaussian envelope
    re    = gauss · cos(w0 · t / s)      # real (cosine-modulated)
    im    = gauss · sin(w0 · t / s)      # imaginary (sine-modulated)

Output: fig10_morlet_anatomy.png — 16:9, 300 DPI, presentation-scale fonts.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

C_ENV      = "#444444"   # neutral gray for envelope
C_RE       = "#21918c"   # teal for real (cos) part — line color
C_IM       = "#440154"   # purple for imaginary (sin) part — line color
C_RE_TEXT  = "#0f5a55"   # darker teal for text on white
C_IM_TEXT  = "#2a0a3a"   # darker purple for text on white

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


def main() -> None:
    # parameters chosen for clear visualization: enough cycles to see the
    # oscillation, few enough that they stay visually distinct on the slide.
    # at w0=6, the wavelet has ~5 visible cycles inside ±3σ of its envelope.
    w0 = 6.0
    f = 20.0           # Hz — slow enough to count cycles by eye
    sr = 2000.0        # samples per second (visualization only)
    s = w0 / (2 * np.pi * f)
    duration = 8.0 * s              # ±4σ on either side of t=0
    n = int(round(duration * sr)) | 1
    t = (np.arange(n) - n // 2) / sr

    norm = (np.pi * s) ** -0.25
    gauss = norm * np.exp(-0.5 * (t / s) ** 2)
    re = gauss * np.cos(w0 * t / s)
    im = gauss * np.sin(w0 * t / s)

    # Renormalize each panel's curves so peak = 1.0; same y-scale across panels.
    gauss_n = gauss / np.max(gauss)
    re_n    = re    / np.max(np.abs(re))
    im_n    = im    / np.max(np.abs(im))

    t_ms = t * 1000.0
    y_max = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0))

    # ---- panel 1: Gaussian envelope -----------------------------------------
    ax = axes[0]
    ax.plot(t_ms, gauss_n, color=C_ENV)
    ax.fill_between(t_ms, gauss_n, 0, color=C_ENV, alpha=0.10)
    ax.set_title("1.  Gaussian envelope", fontweight="bold", pad=12)
    ax.set_xlabel("time  (ms)")
    ax.set_ylabel("amplitude")
    ax.set_ylim(-y_max, y_max)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.text(0.04, 0.93,
            r"$(\pi s)^{-1/4} \cdot e^{-\frac{1}{2}(t/s)^2}$",
            transform=ax.transAxes, fontsize=16, color=C_ENV, va="top")

    # ---- panel 2: complex sinusoid (zoomed to ±2 cycles for legibility) ----
    ax = axes[1]
    ax.plot(t_ms, np.cos(w0 * t / s), color=C_RE,
            label=r"$\cos(w_0 t / s)$")
    ax.plot(t_ms, np.sin(w0 * t / s), color=C_IM,
            label=r"$\sin(w_0 t / s)$", linestyle="--")
    ax.set_title("2.  Complex sinusoid", fontweight="bold", pad=12)
    ax.set_xlabel("time  (ms)")
    ax.set_ylim(-y_max, y_max)
    period_ms = 1000.0 / f
    ax.set_xlim(-2.0 * period_ms, 2.0 * period_ms)   # show ~4 cycles
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    leg = ax.legend(frameon=False, loc="upper right")
    for text, c in zip(leg.get_texts(), [C_RE_TEXT, C_IM_TEXT]):
        text.set_color(c)

    # ---- panel 3: their product = the Morlet wavelet -----------------------
    ax = axes[2]
    ax.plot(t_ms, gauss_n, color=C_ENV, linestyle=":", linewidth=1.6,
            label="envelope")
    ax.plot(t_ms, -gauss_n, color=C_ENV, linestyle=":", linewidth=1.6)
    ax.plot(t_ms, re_n, color=C_RE, label="real part")
    ax.plot(t_ms, im_n, color=C_IM, label="imag part", linestyle="--")
    ax.set_title("3.  Morlet wavelet  =  1 × 2", fontweight="bold", pad=12)
    ax.set_xlabel("time  (ms)")
    ax.set_ylim(-y_max, y_max)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    leg = ax.legend(frameon=False, loc="upper right", ncol=1)
    for text, c in zip(leg.get_texts(), [C_ENV, C_RE_TEXT, C_IM_TEXT]):
        text.set_color(c)

    # parameter footer (under the panels)
    fig.text(0.5, -0.02,
             rf"example: $w_0 = {w0:.0f}$,   $f = {f:.0f}$ Hz,   "
             rf"$s = w_0 / (2\pi f) \approx {s*1000:.2f}$ ms,   "
             rf"sample rate = {int(sr)} Hz",
             ha="center", va="top", fontsize=14, color="#555555",
             family="monospace")

    fig.tight_layout(pad=1.5)
    out = ROOT / "fig10_morlet_anatomy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
