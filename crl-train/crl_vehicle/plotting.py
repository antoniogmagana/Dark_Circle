"""
plotting.py — shared poster-style plotting helpers.

Single source of truth for figure styling, top-N selection, color/linestyle
assignment, and save paths. All scripts under crl-train/ import from here so
poster styling is one rcParams dict instead of scattered overrides.

Public API
----------
    apply_poster_style()         — set matplotlib rcParams to poster defaults
    top_n_runs(rms, n, by, ...)  — rank and truncate to top-N runs
    assign_run_styles(rms)       — per-run color + linestyle + label
    poster_save(fig, path)       — save PNG+PDF, close figure
    plot_confusion_matrix(...)   — single confusion-matrix figure

Constants
---------
    POSTER_RCPARAMS — rcParams dict applied by apply_poster_style()
    POSTER_PALETTE  — 5-class qualitative color palette
    LINESTYLES      — solid / dashed / dashdot / dotted

Design notes
------------
- Color encodes individual run identity (one slot per run, distinct hues).
- Linestyle encodes frontend family (solid/dashed/dashdot/dotted, assigned
  in the order frontends appear in the input). Together, color + linestyle
  give two redundant signals so a poster viewer can identify a line by
  either dimension.
- "Lower is better" metrics (best_val_ref_elbo) are detected by name in
  top_n_runs so callers don't have to flip the sort manually.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from crl_vehicle import analysis as A

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

POSTER_RCPARAMS: dict = {
    "font.size": 16,
    "font.weight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    # Embed fonts in PDF for poster printers (TrueType, not Type-3 bitmaps).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
}


def apply_poster_style() -> None:
    """Apply poster rcParams to matplotlib globally. Idempotent."""
    matplotlib.rcParams.update(POSTER_RCPARAMS)


# --------------------------------------------------------------------------
# Top-N selection
# --------------------------------------------------------------------------

# Fields where lower is better (sort ascending in top_n_runs).
_LOWER_IS_BETTER = {"best_val_ref_elbo", "final_val_recon", "final_val_raw_kl", "final_val_loss"}


def top_n_runs(
    rms: list[A.RunMetrics],
    n: int = 5,
    by: str = "best_type_f1",
    exclude_diverged: bool = True,
) -> list[A.RunMetrics]:
    """Return the top-N runs ranked by `by`.

    Drops runs where the field is None. Drops runs where `diverged=True`
    when `exclude_diverged` is True (default). Auto-flips sort direction
    for known lower-is-better metrics.
    """
    pool = [rm for rm in rms if not (exclude_diverged and rm.diverged)]
    pool = [rm for rm in pool if getattr(rm, by, None) is not None]
    ascending = by in _LOWER_IS_BETTER
    pool.sort(key=lambda rm: getattr(rm, by), reverse=not ascending)
    return pool[:n]


# --------------------------------------------------------------------------
# Color + linestyle assignment
# --------------------------------------------------------------------------

# Five-class qualitative palette. Picked from tab10 in an order that
# maximizes pairwise contrast (blue / orange / green / red / purple).
POSTER_PALETTE: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]

# Four clean built-in linestyles, readable from poster distance.
LINESTYLES: list[str] = ["-", "--", "-.", ":"]


def assign_run_styles(rms: list[A.RunMetrics]) -> list[dict]:
    """Return a list of {color, linestyle, label} parallel to `rms`.

    Returned list is the same length and order as input — callers iterate
    `for rm, s in zip(rms, styles)`. Returning a list (rather than a
    dict keyed by name) avoids collisions when two runs share a basename
    (e.g. both saved under different parent dirs as `v3_lowfreq`).

    color: distinct slot from POSTER_PALETTE in input order (caller should
        pass already-sorted top-N).
    linestyle: assigned by frontend family, in the order families appear.
        Wraps if more than len(LINESTYLES) families are present.
    label: "<run.name> (<frontend_type>)" — both signals appear in legend.
    """
    family_to_ls: dict[str, str] = {}
    out: list[dict] = []
    for i, rm in enumerate(rms):
        family = A._FRONTEND_FAMILY.get(rm.config.get("frontend_type", ""), "other")
        if family not in family_to_ls:
            family_to_ls[family] = LINESTYLES[len(family_to_ls) % len(LINESTYLES)]
        out.append(
            {
                "color": POSTER_PALETTE[i % len(POSTER_PALETTE)],
                "linestyle": family_to_ls[family],
                "label": f"{rm.name} ({rm.config.get('frontend_type', '?')})",
            }
        )
    return out


# --------------------------------------------------------------------------
# Save helper
# --------------------------------------------------------------------------


def poster_save(fig, out_stem: Path) -> None:
    """Save fig to PNG + PDF at out_stem (no suffix), close fig.

    DPI comes from rcParams (apply_poster_style sets it to 300). Creates
    parent dir if needed.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")


# --------------------------------------------------------------------------
# Confusion matrix (factored out of eval.py)
# --------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    title: str,
    out_path: Path,
    cmap: str = "Blues",
) -> None:
    """Render a single normalized (per-row) confusion matrix.

    Cell text shows raw count and row-percent. Cell text color flips to
    white above 0.6 row-fraction for legibility on dark cells.
    Inherits font sizes from rcParams (set by apply_poster_style).
    """
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = cm_arr / np.where(row_sums == 0, 1, row_sums)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.4 + 1.5), max(5, n * 1.4)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            count = int(cm_arr[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.0%})",
                ha="center",
                va="center",
                fontweight="bold",
                color=color,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
