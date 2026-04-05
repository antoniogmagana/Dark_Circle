"""
file_parse.py
=============
Reads raw sensor data files (CSV/parquet), applies the three-gate vehicle-presence
labeling logic from sample_parse.py, and writes one flat parquet file per logical
DB table.

Output file naming mirrors the PostgreSQL table convention:
    {dataset}_{signal}_{vehicle}_{sensor}.parquet

Each file contains:
    time_stamp      — relative seconds from recording start (matches DB)
    unix_timestamp  — wall-clock Unix seconds (from GPS/timestamps/parquet metadata)
    signal columns  — amplitude, or accel_x_ew/accel_y_ns/accel_z_ud
    present         — vehicle-presence label (bool)
    scene_id / run_id — m3nvc only

sample_id is omitted; PostgreSQL BIGSERIAL assigns it at reload time.

Usage
-----
    python file_parse.py --output-dir ./parsed_data
    python file_parse.py --datasets iobt focal --output-dir ./test_out
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import variables
from sample_parse import bandpass, label_windows, AUDIO_BAND, BASE_RATES, M3NVC_RATES
from load_db import sanitize_name, determine_dataset


def pad_labels(present_arr: np.ndarray, target_length: int) -> np.ndarray:
    """Pads the window labels out to the exact number of raw samples."""
    if len(present_arr) >= target_length:
        return present_arr[:target_length]

    pad_val = present_arr[-1] if len(present_arr) > 0 else False
    return np.concatenate(
        [present_arr, np.full(target_length - len(present_arr), pad_val)]
    )


# ============================================================
# Output helper
# ============================================================


def write_parquet(
    df: pd.DataFrame,
    dataset: str,
    signal: str,
    vehicle: str,
    sensor: str,
    output_dir: str,
) -> None:
    """Write (or append to) the parquet file for this dataset/signal/vehicle/sensor."""
    table_name = f"{dataset}_{signal}_{sanitize_name(vehicle)}_{sanitize_name(sensor)}"
    out_path = Path(output_dir) / f"{table_name}.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"  → {out_path.name}  ({len(df):,} rows)")


# ============================================================
# IOBT / FOCAL processors
# ============================================================


def _get_start_timestamp(sensor_dir: Path, dataset: str) -> float:
    """Return the Unix timestamp for the first sample in this sensor directory.

    For iobt: ehz.csv col 1 is preferred because it timestamps the actual
    recording start; gps.csv may begin logging seconds later.
    For focal: timestamps.csv provides per-sample Unix times.
    """
    if dataset == "iobt":
        # Use the first seismic timestamp as the anchor — it reflects the true
        # recording start, unlike gps.csv which can lag by tens of seconds.
        ehz_path = sensor_dir / "ehz.csv"
        if ehz_path.exists():
            row = pd.read_csv(ehz_path, sep=r"\s+", header=None, nrows=1)
            if row.shape[1] >= 2:
                return float(row.iloc[0, 1])
        # Fall back to gps.csv if seismic timestamps unavailable
        gps_path = sensor_dir / "gps.csv"
        if gps_path.exists():
            row = pd.read_csv(gps_path, header=None, nrows=1)
            return float(row.iloc[0, 0])
        return 0.0
    else:  # focal
        ts_path = sensor_dir / "timestamps.csv"
        if ts_path.exists():
            row = pd.read_csv(ts_path, header=None, nrows=1)
            return float(row.iloc[0, 0])
        return 0.0


def process_iobt_focal_audio(
    sensor_dir: Path, dataset: str, vehicle: str, sensor: str
) -> pd.DataFrame | None:
    """Read audio CSV, apply bandpass + label_windows, return labeled DataFrame."""
    audio_candidates = ["aud16000.csv", "aud160000.csv", "aud.csv"]
    audio_path = next(
        (sensor_dir / f for f in audio_candidates if (sensor_dir / f).exists()), None
    )
    if audio_path is None:
        return None

    print(f"  Audio  {dataset}/{vehicle}/{sensor}: {audio_path.name}")
    df = pd.read_csv(
        audio_path,
        header=None,
        names=["amplitude"],
        usecols=[0],
        dtype={"amplitude": "float32"},
    )

    if df.empty:
        return None

    n = len(df)
    raw = df["amplitude"].values
    amplitudes = bandpass(raw, *AUDIO_BAND, BASE_RATES["audio"])

    window_size = BASE_RATES["audio"]
    labels = label_windows(
        amplitudes, BASE_RATES["audio"], "audio", raw_signal=raw
    )

    present = np.repeat(labels, window_size)
    present = pad_labels(present, n)

    start_ts = _get_start_timestamp(sensor_dir, dataset)
    time_stamp = np.arange(n, dtype="float32") * variables.ACOUSTIC_PR
    # float64 required — Unix timestamps ~1.6e9 lose sub-second precision in float32
    unix_timestamp = start_ts + np.arange(n, dtype="float64") * variables.ACOUSTIC_PR

    return pd.DataFrame(
        {
            "time_stamp": time_stamp,
            "unix_timestamp": unix_timestamp,
            "amplitude": df["amplitude"].values,
            "present": present,
        }
    )


def process_iobt_focal_seismic(
    sensor_dir: Path, dataset: str, vehicle: str, sensor: str
) -> pd.DataFrame | None:
    """Read ehz.csv (amp + batch timestamp), apply label_windows."""
    seismic_path = sensor_dir / "ehz.csv"
    if not seismic_path.exists():
        return None

    print(f"  Seismic {dataset}/{vehicle}/{sensor}: ehz.csv")
    raw = pd.read_csv(seismic_path, sep=r"\s+", header=None)

    amplitudes = raw.iloc[:, 0].values.astype("float32")
    n = len(amplitudes)

    window_size = BASE_RATES["seismic"]
    labels = label_windows(amplitudes, BASE_RATES["seismic"], "seismic")

    present = np.repeat(labels, window_size)
    present = pad_labels(present, n)

    time_stamp = np.arange(n, dtype="float32") * variables.SEISMIC_PR

    # Use batch timestamps from file if present, otherwise derive from start
    if raw.shape[1] >= 2:
        unix_timestamp = raw.iloc[:, 1].values.astype("float64")
    else:
        start_ts = _get_start_timestamp(sensor_dir, dataset)
        unix_timestamp = start_ts + time_stamp

    return pd.DataFrame(
        {
            "time_stamp": time_stamp,
            "unix_timestamp": unix_timestamp,
            "amplitude": amplitudes,
            "present": present,
        }
    )


def process_iobt_focal_accel(
    sensor_dir: Path, dataset: str, vehicle: str, sensor: str
) -> pd.DataFrame | None:
    """Read ene/enn/enz CSV triplet, compute magnitude, apply label_windows."""
    accel_files = {
        "accel_x_ew": sensor_dir / "ene.csv",
        "accel_y_ns": sensor_dir / "enn.csv",
        "accel_z_ud": sensor_dir / "enz.csv",
    }
    if not all(p.exists() for p in accel_files.values()):
        return None

    print(f"  Accel  {dataset}/{vehicle}/{sensor}")
    axes = {
        col: pd.read_csv(p, header=None, names=[col])[col].values.astype("float32")
        for col, p in accel_files.items()
    }

    n = min(len(v) for v in axes.values())
    x = axes["accel_x_ew"][:n]
    y = axes["accel_y_ns"][:n]
    z = axes["accel_z_ud"][:n]
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    window_size = BASE_RATES["seismic"]
    labels = label_windows(magnitude, BASE_RATES["seismic"], "accel")

    present = np.repeat(labels, window_size)
    present = pad_labels(present, n)

    time_stamp = np.arange(n, dtype="float32") * variables.SEISMIC_PR

    ts_path = sensor_dir / "timestamps.csv"
    if ts_path.exists():
        ts_raw = pd.read_csv(ts_path, header=None).iloc[:n, 0].values.astype("float64")
        unix_timestamp = (
            ts_raw
            if len(ts_raw) == n
            else np.pad(ts_raw, (0, n - len(ts_raw)), mode="edge")
        )
    else:
        start_ts = _get_start_timestamp(sensor_dir, dataset)
        unix_timestamp = start_ts + time_stamp

    return pd.DataFrame(
        {
            "time_stamp": time_stamp,
            "unix_timestamp": unix_timestamp,
            "accel_x_ew": x,
            "accel_y_ns": y,
            "accel_z_ud": z,
            "present": present,
        }
    )


# ============================================================
# IOBT / FOCAL walker
# ============================================================


def process_mod_vehicle(root_dir: Path, output_dir: str) -> None:
    """Walk MOD_vehicle directory structure, process all sensor folders."""
    visited = set()

    for sensor_dir in sorted(root_dir.rglob("*")):
        if not sensor_dir.is_dir():
            continue
        if sensor_dir in visited:
            continue

        dataset = determine_dataset(sensor_dir)
        if not dataset:
            continue

        visited.add(sensor_dir)
        vehicle = sensor_dir.parent.name
        sensor = sensor_dir.name

        for process_fn, signal in [
            (process_iobt_focal_audio, "audio"),
            (process_iobt_focal_seismic, "seismic"),
            (process_iobt_focal_accel, "accel"),
        ]:
            df = process_fn(sensor_dir, dataset, vehicle, sensor)
            if df is not None:
                write_parquet(df, dataset, signal, vehicle, sensor, output_dir)


# ============================================================
# M3NVC processor
# ============================================================

# Local scene_id registry (mirrors load_db.py get_scene_id without needing a DB)
_scene_registry: dict[str, int] = {}


def _local_scene_id(scene_name: str) -> int:
    if scene_name not in _scene_registry:
        _scene_registry[scene_name] = len(_scene_registry) + 1
    return _scene_registry[scene_name]


def process_m3nvc_scene(scene_dir: Path, output_dir: str) -> None:
    """Process all run parquets in one M3NVC scene directory."""
    sc_id = _local_scene_id(scene_dir.name)

    # Build run_id → vehicle_label map
    run_map: dict[int, str] = {}
    meta_path = scene_dir / "run_ids.parquet"
    if meta_path.exists():
        try:
            meta_df = pd.read_parquet(meta_path)
            meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]

            def _clean(val):
                if isinstance(val, (list, np.ndarray)):
                    return "+".join(str(v) for v in val)
                return val

            if "label" in meta_df.columns:
                meta_df["label"] = meta_df["label"].apply(_clean)
            meta_df = meta_df.drop_duplicates(subset=["run_id"])
            run_map = meta_df.set_index("run_id")["label"].to_dict()
        except Exception as e:
            print(f"  [m3nvc] Could not load run_ids for {scene_dir.name}: {e}")

    # Accumulate per (signal, vehicle, sensor) before writing
    # so multiple runs of the same vehicle concat into one file
    accum: dict[tuple, list[pd.DataFrame]] = {}

    for p_file in sorted(scene_dir.glob("*.parquet")):
        fname = p_file.name
        if fname in ("run_ids.parquet", "sensor_location.parquet"):
            continue
        if "gps" in fname or "dis" in fname:
            continue

        match = re.match(r"run(\d+)_rs(\d+)_([a-z]{3})\.parquet", fname)
        if not match:
            continue

        run_num = int(match.group(1))
        sensor_node = f"rs{match.group(2)}"
        modality = match.group(3)

        if modality not in ("mic", "geo"):
            continue

        if run_num in (8, 9):
            vehicle_label = "background"
        else:
            vehicle_label = run_map.get(run_num)
            if not vehicle_label:
                continue

        signal_type = "audio" if modality == "mic" else "seismic"
        sample_period = (
            variables.ACOUSTIC_PR2 if modality == "mic" else variables.SEISMIC_PR2
        )

        try:
            df = pd.read_parquet(p_file)
            if df.empty:
                continue

            df = df.sort_values(df.columns[0])  # sort by first column (timestamp)

            # Extract unix_timestamp
            ts_col = df.columns[0]
            raw_ts = df[ts_col].values
            if np.issubdtype(raw_ts.dtype, np.datetime64):
                unix_ts = raw_ts.astype("datetime64[ns]").astype("int64") / 1e9
            else:
                unix_ts = raw_ts.astype("float64")

            n = len(df)
            time_stamp = np.arange(n, dtype="float32") * sample_period

            if modality == "mic":
                amp_col = "samples" if "samples" in df.columns else df.columns[-1]
                amplitudes = df[amp_col].values.astype("float32")
                amplitudes_filt = bandpass(
                    amplitudes, *AUDIO_BAND, M3NVC_RATES["audio"]
                )
                labels = label_windows(
                    amplitudes_filt,
                    M3NVC_RATES["audio"],
                    "audio",
                    raw_signal=amplitudes,
                )
                window_size = M3NVC_RATES["audio"]
                present = np.repeat(labels, window_size)
                present = pad_labels(present, n)

                out_df = pd.DataFrame(
                    {
                        "scene_id": sc_id,
                        "run_id": run_num,
                        "time_stamp": time_stamp,
                        "unix_timestamp": unix_ts,
                        "amplitude": amplitudes,
                        "present": present,
                    }
                )

            else:  # geo — may be long-format (timestamp, channel, samples)
                if "channel" in df.columns:
                    amp_col = "samples" if "samples" in df.columns else "amplitude"
                    channel_col = df["channel"].values
                    amplitudes = df[amp_col].values.astype("float32")
                else:
                    amp_col = [c for c in df.columns if c != ts_col][0]
                    amplitudes = df[amp_col].values.astype("float32")
                    channel_col = np.full(n, "UD")

                labels = label_windows(amplitudes, M3NVC_RATES["seismic"], "seismic")
                window_size = M3NVC_RATES["seismic"]
                present = np.repeat(labels, window_size)
                present = pad_labels(present, n)

                out_df = pd.DataFrame(
                    {
                        "scene_id": sc_id,
                        "run_id": run_num,
                        "time_stamp": time_stamp,
                        "unix_timestamp": unix_ts,
                        "channel": channel_col,
                        "amplitude": amplitudes,
                        "present": present,
                    }
                )

            key = (signal_type, vehicle_label, sensor_node)
            accum.setdefault(key, []).append(out_df)
            print(
                f"  [m3nvc] {scene_dir.name} run{run_num} {signal_type} {sensor_node}"
                f" ({vehicle_label})  {n:,} rows"
            )

        except Exception as e:
            print(f"  [m3nvc] Error processing {fname}: {e}")

    # Write accumulated runs
    for (signal_type, vehicle_label, sensor_node), dfs in accum.items():
        combined = pd.concat(dfs, ignore_index=True)
        write_parquet(
            combined, "m3nvc", signal_type, vehicle_label, sensor_node, output_dir
        )


def process_m3nvc(root_path: str, output_dir: str) -> None:
    root = Path(root_path).expanduser()
    if not root.exists():
        print(f"M3NVC path {root} does not exist. Skipping.")
        return
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            print(f"\n[m3nvc] Scene: {scene_dir.name}")
            process_m3nvc_scene(scene_dir, output_dir)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse raw sensor files and write labeled parquet files."
    )
    parser.add_argument(
        "--output-dir",
        default="./parsed_data",
        help="Directory to write output parquet files (default: ./parsed_data)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["iobt", "focal", "m3nvc", "all"],
        default=["all"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--mod-dir",
        default=None,
        help="Override MOD_vehicle data directory (default: variables.MOD_path)",
    )
    parser.add_argument(
        "--m3nvc-dir",
        default=None,
        help="Override M3NVC data directory (default: variables.M3NVC_path)",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    requested = set(args.datasets)
    datasets = {"iobt", "focal", "m3nvc"} if "all" in requested else requested

    if datasets & {"iobt", "focal"}:
        root = (
            Path(args.mod_dir).expanduser()
            if args.mod_dir
            else Path(variables.MOD_path).expanduser()
        )
        print(f"\n--- Processing IOBT & FOCAL from {root} ---")
        process_mod_vehicle(root, args.output_dir)

    if "m3nvc" in datasets:
        m3nvc_root = args.m3nvc_dir or variables.M3NVC_path
        print(f"\n--- Processing M3NVC from {m3nvc_root} ---")
        process_m3nvc(m3nvc_root, args.output_dir)

    print("\nDone. Output files:")
    for p in sorted(Path(args.output_dir).glob("*.parquet")):
        size_mb = p.stat().st_size / 1_048_576
        print(f"  {p.name}  ({size_mb:.1f} MB)")
