"""
analyze_data.py
===============
Generates charts and statistics from labeled parquet files produced by file_parse.py.

Usage
-----
    poetry run python analyze_data.py --data-dir ../data/parsed --output-dir ./charts

Produces
--------
    fig1_timeseries.png       — Seismic + audio time series with vehicle presence overlay
    fig2_presence_counts.png  — Labeled sample counts by vehicle and modality
    fig3_rms_distributions.png — Per-window RMS: vehicle vs background
    fig4_spectral_density.png  — Mean PSD: vehicle vs background (seismic)
    fig5_noise_floors.png      — Noise floor and SNR by recording
    statistics.txt             — Full dataset statistics
"""

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
})

VEHICLE_COLOR  = "#2563EB"   # blue  — vehicle present
BACKGROUND_COLOR = "#94A3B8" # slate — background
AUDIO_COLOR    = "#059669"   # green
SEISMIC_COLOR  = "#DC2626"   # red


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_rate(fname: str) -> int:
    return 16000 if "audio" in fname else 100


def _window_rms(df: pd.DataFrame, sr: int) -> pd.DataFrame:
    """Return per-1-second-window AC RMS and present label."""
    amp = df["amplitude"].values.astype("float64")
    n_win = len(amp) // sr
    wins = amp[: n_win * sr].reshape(n_win, sr)
    centered = wins - wins.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=1))

    pres = df["present"].values
    pres_win = np.array([pres[i * sr] for i in range(n_win)])

    ts = df["time_stamp"].values
    ts_win = ts[: n_win * sr : sr]

    return pd.DataFrame({"time_stamp": ts_win, "rms": rms, "present": pres_win})


def _pretty_name(fname: str) -> str:
    """iobt_audio_polaris0150pm_rs1 → Polaris 0150pm"""
    parts = fname.replace(".parquet", "").split("_")
    # drop dataset and signal prefix; drop trailing sensor (rsN)
    middle = parts[2:-1]
    return " ".join(p.capitalize() for p in middle)


def _load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    print(f"Loading {len(files)} parquet files from {data_dir} …")
    out = {}
    for p in files:
        out[p.name] = pd.read_parquet(p)
        print(f"  {p.name}  ({len(out[p.name]):,} rows)")
    return out


# ── Figure 1: Time series with presence overlay ───────────────────────────────

def fig_timeseries(data: dict, out_dir: Path) -> None:
    # Pick one seismic and one audio file
    seismic_files = [k for k in data if "seismic" in k]
    audio_files   = [k for k in data if "audio"   in k]
    if not seismic_files or not audio_files:
        print("Skipping fig1: need at least one seismic and one audio file.")
        return

    s_key = seismic_files[0]
    a_key = next((k for k in audio_files if s_key.split("_seismic_")[1] in k), audio_files[0])

    s_df = data[s_key]
    a_df = data[a_key]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(
        f"Vehicle Detection — {_pretty_name(s_key)} (IOBT)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ax, df, sr, label, color in [
        (axes[0], s_df, 100,   "Seismic (100 Hz)",  SEISMIC_COLOR),
        (axes[1], a_df, 16000, "Audio (16 kHz)",     AUDIO_COLOR),
    ]:
        t = df["time_stamp"].values
        amp = df["amplitude"].values
        pres = df["present"].values

        # Downsample for plotting (max 10k points)
        step = max(1, len(t) // 10000)
        ax.plot(t[::step], amp[::step], lw=0.5, color=color, alpha=0.7, label=label)

        # Shade vehicle-present windows
        in_vehicle = False
        win_start = None
        for i in range(0, len(pres) - sr, sr):
            if pres[i] and not in_vehicle:
                win_start = t[i]
                in_vehicle = True
            elif not pres[i] and in_vehicle:
                ax.axvspan(win_start, t[i], color=VEHICLE_COLOR, alpha=0.15, lw=0)
                in_vehicle = False
        if in_vehicle:
            ax.axvspan(win_start, t[-1], color=VEHICLE_COLOR, alpha=0.15, lw=0)

        pct = pres.mean() * 100
        ax.set_ylabel("Amplitude (ADC counts)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{label}   —   {pct:.1f}% vehicle-present", fontsize=10)

    vehicle_patch = mpatches.Patch(color=VEHICLE_COLOR, alpha=0.4, label="Vehicle present")
    fig.legend(handles=[vehicle_patch], loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_timeseries.png")
    plt.close(fig)
    print("  Saved fig1_timeseries.png")


# ── Figure 2: Present vs background sample counts ────────────────────────────

def fig_presence_counts(data: dict, out_dir: Path) -> None:
    rows = []
    for fname, df in data.items():
        parts = fname.replace(".parquet", "").split("_")
        modality = parts[1]       # audio / seismic
        vehicle  = _pretty_name(fname)
        n_present    = df["present"].sum()
        n_background = (~df["present"]).sum()
        rows.append({"vehicle": vehicle, "modality": modality,
                     "present": n_present, "background": n_background})

    meta = pd.DataFrame(rows)

    for mod in ["seismic", "audio"]:
        sub = meta[meta["modality"] == mod].sort_values("vehicle")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(sub))
        w = 0.45
        b1 = ax.bar(x - w/2, sub["background"] / 1e6, w,
                    color=BACKGROUND_COLOR, label="Background")
        b2 = ax.bar(x + w/2, sub["present"] / 1e6,    w,
                    color=VEHICLE_COLOR,   label="Vehicle present")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["vehicle"], rotation=30, ha="right")
        ax.set_ylabel("Samples (millions)")
        ax.set_title(
            f"Vehicle vs Background Samples — {mod.capitalize()} (IOBT)",
            fontsize=12, fontweight="bold",
        )
        ax.legend()

        # Annotate % present
        for i, (_, row) in enumerate(sub.iterrows()):
            pct = row["present"] / (row["present"] + row["background"]) * 100
            ax.text(i + w/2, row["present"] / 1e6 + 0.05, f"{pct:.0f}%",
                    ha="center", va="bottom", fontsize=8, color=VEHICLE_COLOR)

        fig.tight_layout()
        fname_out = f"fig2_presence_counts_{mod}.png"
        fig.savefig(out_dir / fname_out)
        plt.close(fig)
        print(f"  Saved {fname_out}")


# ── Figure 3: Per-window RMS distributions ───────────────────────────────────

def fig_rms_distributions(data: dict, out_dir: Path) -> None:
    for mod, sr in [("seismic", 100), ("audio", 16000)]:
        mod_files = {k: v for k, v in data.items() if mod in k}
        if not mod_files:
            continue

        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
        fig.suptitle(
            f"Per-Window RMS Distribution — {mod.capitalize()} (IOBT)\n"
            "Blue = vehicle present   Grey = background",
            fontsize=12, fontweight="bold",
        )
        axes_flat = axes.flatten()

        for ax, (fname, df) in zip(axes_flat, mod_files.items()):
            wdf = _window_rms(df, sr)
            bg  = wdf.loc[~wdf["present"], "rms"]
            veh = wdf.loc[ wdf["present"], "rms"]

            bins = np.linspace(0, wdf["rms"].quantile(0.99), 50)
            ax.hist(bg,  bins=bins, color=BACKGROUND_COLOR, alpha=0.7, density=True, label="Background")
            ax.hist(veh, bins=bins, color=VEHICLE_COLOR,    alpha=0.7, density=True, label="Vehicle")
            ax.set_title(_pretty_name(fname), fontsize=9)
            ax.set_xlabel("AC RMS (counts)")
            ax.set_ylabel("Density")

        # Hide unused subplots
        for ax in axes_flat[len(mod_files):]:
            ax.set_visible(False)

        fig.tight_layout()
        fname_out = f"fig3_rms_distributions_{mod}.png"
        fig.savefig(out_dir / fname_out)
        plt.close(fig)
        print(f"  Saved {fname_out}")


# ── Figure 4: Mean power spectral density ────────────────────────────────────

def fig_spectral_density(data: dict, out_dir: Path) -> None:
    seismic_files = {k: v for k, v in data.items() if "seismic" in k}
    if not seismic_files:
        print("Skipping fig4: no seismic files.")
        return

    sr = 100
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    fig.suptitle(
        "Mean Power Spectral Density — Seismic (IOBT)\n"
        "Blue = vehicle present   Grey = background",
        fontsize=12, fontweight="bold",
    )
    axes_flat = axes.flatten()
    freqs = np.fft.rfftfreq(sr, d=1.0 / sr)

    for ax, (fname, df) in zip(axes_flat, seismic_files.items()):
        amp  = df["amplitude"].values.astype("float64")
        pres = df["present"].values
        n_win = len(amp) // sr
        wins = amp[: n_win * sr].reshape(n_win, sr)
        centered = wins - wins.mean(axis=1, keepdims=True)
        pres_win = np.array([pres[i * sr] for i in range(n_win)])

        hann = np.hanning(sr)
        psd = (np.abs(np.fft.rfft(centered * hann, axis=1)) ** 2) / sr

        bg_psd  = psd[~pres_win].mean(axis=0) if (~pres_win).any() else None
        veh_psd = psd[ pres_win].mean(axis=0) if  pres_win.any()  else None

        if bg_psd is not None:
            ax.semilogy(freqs, bg_psd,  color=BACKGROUND_COLOR, lw=1.5, label="Background")
        if veh_psd is not None:
            ax.semilogy(freqs, veh_psd, color=VEHICLE_COLOR,    lw=1.5, label="Vehicle")

        ax.set_title(_pretty_name(fname), fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_xlim(0, 50)

    for ax in axes_flat[len(seismic_files):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_spectral_density.png")
    plt.close(fig)
    print("  Saved fig4_spectral_density.png")


# ── Figure 5: Noise floor and SNR by recording ───────────────────────────────

def fig_noise_floors(data: dict, out_dir: Path) -> None:
    rows = []
    for fname, df in data.items():
        parts = fname.replace(".parquet", "").split("_")
        modality = parts[1]
        sr = _sample_rate(fname)
        wdf = _window_rms(df, sr)

        noise_floor = np.percentile(wdf["rms"], 10)
        bg_rms  = wdf.loc[~wdf["present"], "rms"].median() if (~wdf["present"]).any() else np.nan
        veh_rms = wdf.loc[ wdf["present"], "rms"].median() if  wdf["present"].any()  else np.nan
        snr_db  = 20 * np.log10(veh_rms / noise_floor) if (veh_rms and noise_floor > 0) else np.nan

        rows.append({
            "label":      _pretty_name(fname),
            "modality":   modality,
            "noise_floor": noise_floor,
            "bg_rms":     bg_rms,
            "veh_rms":    veh_rms,
            "snr_db":     snr_db,
        })

    meta = pd.DataFrame(rows)

    for mod in ["seismic", "audio"]:
        sub = meta[meta["modality"] == mod].sort_values("label")
        if sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Noise Floor & SNR by Recording — {mod.capitalize()} (IOBT)",
            fontsize=12, fontweight="bold",
        )

        x = np.arange(len(sub))
        # Noise floor panel
        ax1.bar(x, sub["bg_rms"],     color=BACKGROUND_COLOR, label="Background median RMS")
        ax1.bar(x, sub["veh_rms"],    color=VEHICLE_COLOR,    alpha=0.6, label="Vehicle median RMS")
        ax1.plot(x, sub["noise_floor"], "k--", marker="o", ms=5, lw=1.5, label="Noise floor (10th pctile)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub["label"], rotation=30, ha="right")
        ax1.set_ylabel("AC RMS (ADC counts)")
        ax1.set_title("RMS Levels")
        ax1.legend(fontsize=8)

        # SNR panel
        colors = [VEHICLE_COLOR if v >= 6 else "#F59E0B" for v in sub["snr_db"]]
        ax2.bar(x, sub["snr_db"], color=colors)
        ax2.axhline(6,  color="k", lw=1, ls="--", label="6 dB threshold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(sub["label"], rotation=30, ha="right")
        ax2.set_ylabel("Vehicle SNR (dB above noise floor)")
        ax2.set_title("Vehicle Signal-to-Noise Ratio")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fname_out = f"fig5_noise_floors_{mod}.png"
        fig.savefig(out_dir / fname_out)
        plt.close(fig)
        print(f"  Saved {fname_out}")


# ── Statistics report ─────────────────────────────────────────────────────────

def print_statistics(data: dict, out_dir: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("DARK CIRCLE DATASET STATISTICS")
    lines.append("=" * 72)

    total_samples = sum(len(df) for df in data.values())
    total_present = sum(df["present"].sum() for df in data.values())
    total_bg      = total_samples - total_present

    lines.append(f"\nTotal samples:        {total_samples:>15,}")
    lines.append(f"Vehicle present:      {total_present:>15,}  ({total_present/total_samples*100:.1f}%)")
    lines.append(f"Background:           {total_bg:>15,}  ({total_bg/total_samples*100:.1f}%)")

    for mod, sr in [("seismic", 100), ("audio", 16000)]:
        mod_data = {k: v for k, v in data.items() if mod in k}
        if not mod_data:
            continue

        lines.append(f"\n{'─'*72}")
        lines.append(f"  {mod.upper()}")
        lines.append(f"{'─'*72}")
        lines.append(f"  {'Recording':<38} {'Windows':>7} {'Present':>8} {'Bg':>8} {'%Pres':>6} {'SNR dB':>7} {'NF (RMS)':>9}")
        lines.append(f"  {'─'*38} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*9}")

        for fname, df in sorted(mod_data.items()):
            wdf = _window_rms(df, sr)
            n_win   = len(wdf)
            n_pres  = wdf["present"].sum()
            n_bg    = n_win - n_pres
            pct     = n_pres / n_win * 100 if n_win else 0
            nf      = np.percentile(wdf["rms"], 10)
            veh_rms = wdf.loc[wdf["present"], "rms"].median() if wdf["present"].any() else np.nan
            snr     = 20 * np.log10(veh_rms / nf) if (not np.isnan(veh_rms) and nf > 0) else np.nan
            snr_str = f"{snr:7.1f}" if not np.isnan(snr) else "    N/A"
            lines.append(
                f"  {_pretty_name(fname):<38} {n_win:>7} {n_pres:>8} {n_bg:>8} {pct:>6.1f} {snr_str} {nf:>9.1f}"
            )

    lines.append(f"\n{'─'*72}")
    lines.append("NOISE SUMMARY")
    lines.append(f"{'─'*72}")
    lines.append(
        "  Noise floor = 10th percentile of per-window AC RMS across all windows.\n"
        "  Vehicle SNR = 20·log10(vehicle_median_RMS / noise_floor).\n"
        "  6 dB threshold separates labelled vehicle from background windows.\n"
        "  Recordings below threshold may indicate weak signatures (no line of sight,\n"
        "  distant vehicle, or sensor placement effects)."
    )

    # Flag any recordings where vehicle SNR < 6 dB
    weak = []
    for fname, df in data.items():
        sr = _sample_rate(fname)
        wdf = _window_rms(df, sr)
        nf = np.percentile(wdf["rms"], 10)
        veh_rms = wdf.loc[wdf["present"], "rms"].median() if wdf["present"].any() else np.nan
        if np.isnan(veh_rms) or nf == 0:
            continue
        snr = 20 * np.log10(veh_rms / nf)
        if snr < 6:
            weak.append(f"  {_pretty_name(fname)} ({fname.split('_')[1]}): {snr:.1f} dB")
    if weak:
        lines.append("\n  Low-SNR recordings (vehicle SNR < 6 dB threshold):")
        lines.extend(weak)
    else:
        lines.append("\n  All recordings exceed the 6 dB vehicle SNR threshold.")

    lines.append("\n" + "=" * 72)

    report = "\n".join(lines)
    print(report)
    (out_dir / "statistics.txt").write_text(report)
    print(f"  Saved statistics.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate charts and statistics from labeled parquet files."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing parquet files from file_parse.py",
    )
    parser.add_argument(
        "--output-dir",
        default="./charts",
        help="Directory to write charts and statistics (default: ./charts)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all(data_dir)

    print("\nGenerating figures …")
    fig_timeseries(data, out_dir)
    fig_presence_counts(data, out_dir)
    fig_rms_distributions(data, out_dir)
    fig_spectral_density(data, out_dir)
    fig_noise_floors(data, out_dir)

    print("\nComputing statistics …")
    print_statistics(data, out_dir)

    print(f"\nAll outputs written to {out_dir}/")
