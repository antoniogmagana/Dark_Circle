import argparse
import sys
import numpy as np
import pandas as pd
import psycopg2
from scipy.signal import butter, sosfiltfilt
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib
import io
import os

import variables

"""Parse vehicle recordings to separate samples of interest from background noise"""

# Constants
AUDIO_BITS = 16
SEISMIC_BITS = 24

BASE_RATES = {"audio": 16000, "seismic": 100}
M3NVC_RATES = {"audio": 1600, "seismic": 200}

# Noise floor: 10th percentile of per-window RMS across all windows
NOISE_FACTOR = 10

SAMPLE_WINDOW = 1.0

# Bandpass for audio only (Hz) — rejects bird calls (>1 kHz) and DC/infrasound
AUDIO_BAND = (10.0, 1000.0)
# Seismic and accel: no bandpass — preserve sensitivity to walking and small signals

# Per-sensor energy gate: dB above noise floor required (Gate 1)
THRESHOLD_DB = {
    "audio":   6.0,
    "seismic": 6.0,
    "accel":   6.0,
}

# Per-sensor persistence: consecutive 1-second windows required (Gate 2)
PERSISTENCE_WIN = {
    "audio":   2,
    "seismic": 2,
    "accel":   2,
}

# Spectral fraction gate — audio only (Gate 3)
# Minimum fraction of energy within AUDIO_BAND required to pass
AUDIO_SPECTRAL_FRAC = 0.30

ROOT = str(".")

USE_FAST_COPY = True


def db_connect():
    try:
        conn = psycopg2.connect(**variables.conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise e
    return conn, cursor


def db_close(conn, cursor):
    cursor.close()
    conn.close()


def get_table_list(conn, cursor):
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE';
    """
    cursor.execute(query)

    return [row[0] for row in cursor.fetchall()]


def bandpass(signal, low, high, fs, order=4):
    """Zero-phase Butterworth bandpass filter for offline processing.

    high is clamped to 49.9 % of fs so the filter remains valid regardless
    of the sample rate (e.g. M3NVC audio at 1600 Hz has Nyquist = 800 Hz,
    which is below the 1000 Hz upper bound of AUDIO_BAND).
    """
    high = min(high, fs * 0.499)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def label_windows(amplitudes, sample_rate, sensor_type, raw_signal=None):
    """Label 1-second windows for vehicle presence using three gates.

    Parameters
    ----------
    amplitudes : array-like
        Signal used for Gates 1 & 2 (may be bandpass-filtered).
    sample_rate : int
        Samples per second.
    sensor_type : str
        One of "audio", "seismic", "accel".
    raw_signal : array-like, optional
        Unfiltered signal used exclusively for Gate 3 spectral fraction check.
        When None, Gate 3 falls back to ``amplitudes`` (pre-filtered), which
        makes the check trivially easy to pass and should be avoided for audio.
    """
    window_size = int(SAMPLE_WINDOW * sample_rate)
    num_windows = len(amplitudes) // window_size
    if num_windows == 0:
        return np.array([], dtype=bool)

    windows = amplitudes[: num_windows * window_size].reshape(num_windows, window_size)

    # Gate 1 — RMS energy: center each window to remove DC offset, then compute RMS.
    # Raw ADC values carry a large constant bias (~16000 counts) that would dominate
    # the RMS and make all windows look equal, masking vehicle-induced variation.
    win_centered = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(win_centered ** 2, axis=1))
    noise_floor = np.percentile(rms, NOISE_FACTOR)
    threshold = noise_floor * (10 ** (THRESHOLD_DB[sensor_type] / 20.0))
    energy_pass = rms >= threshold

    # Gate 2 — Persistence: require PERSISTENCE_WIN consecutive passing windows.
    # Backward-looking: window i passes only if it AND the preceding req-1 windows
    # all passed Gate 1. Left-pad with False so early windows cannot look forward.
    req = PERSISTENCE_WIN[sensor_type]
    padded = np.concatenate([np.zeros(req - 1, dtype=bool), energy_pass])
    rolling_sum = np.convolve(padded.astype(int), np.ones(req, dtype=int), mode="valid")
    persistence_pass = rolling_sum >= req

    # Gate 3 — Spectral fraction: audio only.
    # Must operate on the raw (unfiltered) signal so broadband non-vehicle noise
    # (thunder, HVAC hum) is distinguishable from in-band vehicle energy.
    if sensor_type == "audio":
        spectral_src = raw_signal if raw_signal is not None else amplitudes
        raw_windows = (
            spectral_src[: num_windows * window_size].reshape(num_windows, window_size)
        )
        hann = np.hanning(window_size)
        freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
        # Clamp upper bound to match what bandpass() actually passes at this rate
        band_hi = min(AUDIO_BAND[1], sample_rate * 0.499)
        band_mask = (freqs >= AUDIO_BAND[0]) & (freqs <= band_hi)
        spectral_frac = np.zeros(num_windows)
        for i in range(num_windows):
            x = raw_windows[i] - raw_windows[i].mean()
            power = np.abs(np.fft.rfft(x * hann)) ** 2
            total = power.sum()
            spectral_frac[i] = 0.0 if total == 0 else power[band_mask].sum() / total
        spectral_pass = spectral_frac >= AUDIO_SPECTRAL_FRAC
        return energy_pass & persistence_pass & spectral_pass

    return energy_pass & persistence_pass


def process_audio_table(df):
    raw = df["amplitude"].values
    amplitudes = bandpass(raw, *AUDIO_BAND, BASE_RATES["audio"])
    window_size = BASE_RATES["audio"]
    labels = label_windows(amplitudes, BASE_RATES["audio"], "audio", raw_signal=raw)
    df["present"] = False
    df.iloc[: len(labels) * window_size, df.columns.get_loc("present")] = np.repeat(
        labels, window_size
    )
    return df


def process_seismic_table(df):
    window_size = BASE_RATES["seismic"]
    labels = label_windows(df["amplitude"].values, BASE_RATES["seismic"], "seismic")
    df["present"] = False
    df.iloc[: len(labels) * window_size, df.columns.get_loc("present")] = np.repeat(
        labels, window_size
    )
    return df


def process_accel_table(df):
    magnitude = np.sqrt(
        df["accel_x_ew"] ** 2 + df["accel_y_ns"] ** 2 + df["accel_z_ud"] ** 2
    )
    window_size = BASE_RATES["seismic"]
    labels = label_windows(magnitude.values, BASE_RATES["seismic"], "accel")
    df["present"] = False
    df.iloc[: len(labels) * window_size, df.columns.get_loc("present")] = np.repeat(
        labels, window_size
    )
    return df


def _label_m3nvc_group(group, sample_rate, window_size, sensor_type):
    raw = group["amplitude"].values
    if sensor_type == "audio":
        amplitudes = bandpass(raw, *AUDIO_BAND, sample_rate)
        labels = label_windows(amplitudes, sample_rate, sensor_type, raw_signal=raw)
    else:
        labels = label_windows(raw, sample_rate, sensor_type)
    if len(labels) == 0:
        return pd.Series(False, index=group.index)
    present_arr = np.repeat(labels, window_size)
    if len(present_arr) < len(group):
        present_arr = np.concatenate(
            [present_arr, np.full(len(group) - len(present_arr), present_arr[-1])]
        )
    return pd.Series(present_arr[: len(group)], index=group.index)


def process_m3nvc_audio(df):
    df = df.copy()
    window_size = int(SAMPLE_WINDOW * M3NVC_RATES["audio"])
    df["present"] = df.groupby(["scene_id", "run_id"], group_keys=False).apply(
        lambda g: _label_m3nvc_group(g, M3NVC_RATES["audio"], window_size, "audio")
    )
    return df


def process_m3nvc_seismic(df):
    df = df.copy()
    window_size = int(SAMPLE_WINDOW * M3NVC_RATES["seismic"])
    df["present"] = df.groupby(["scene_id", "run_id"], group_keys=False).apply(
        lambda g: _label_m3nvc_group(g, M3NVC_RATES["seismic"], window_size, "seismic")
    )
    return df


def process_table(table):
    conn, cursor = db_connect()  # conn kept for db_close at end
    # extract table designations
    table_type = table.split("_")
    # Add default false column for vehicle presence
    query = f"""ALTER TABLE {table} 
    ADD COLUMN IF NOT EXISTS present BOOLEAN NOT NULL DEFAULT FALSE;"""
    cursor.execute(query)

    # create dataframe for table
    if table_type[0] in ["iobt", "focal"]:
        if table_type[1] == "accel":
            cols = "sample_id, accel_x_ew, accel_y_ns, accel_z_ud"
        else:
            cols = "sample_id, amplitude"
    elif table_type[0] == "m3nvc":
        cols = "sample_id, scene_id, run_id, amplitude"
    else:
        cols = "*"
    with conn.cursor(name=f"ss_{table}") as ss_cur:
        ss_cur.itersize = 500000
        ss_cur.execute(f"SELECT {cols} FROM {table};")
        rows = ss_cur.fetchall()
        col_names = [desc[0] for desc in ss_cur.description]
    df = pd.DataFrame(rows, columns=col_names)
    update = False

    if table_type[0] in ["iobt", "focal"]:
        if table_type[1] == "audio":
            df = process_audio_table(df)
            update = True
        elif table_type[1] == "seismic":
            df = process_seismic_table(df)
            update = True
        elif table_type[1] == "accel":
            df = process_accel_table(df)
            update = True
        else:
            print(f"table {table} schema not found")
            update = True
    elif table_type[0] == "m3nvc":
        if table_type[1] == "audio":
            df = process_m3nvc_audio(df)
            update = True
        elif table_type[1] == "seismic":
            df = process_m3nvc_seismic(df)
            update = True
        else:
            print(f"table {table} schema not found")
    else:
        print(f"table {table} schema not found")

    if update:
        if USE_FAST_COPY:
            buf = io.StringIO(
                df[["sample_id", "present"]].to_csv(sep="\t", header=False, index=False)
            )
        else:
            buf = io.StringIO()
            for sid, pres in zip(df["sample_id"], df["present"]):
                buf.write(f"{sid}\t{pres}\n")
            buf.seek(0)
        cursor.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _present_update "
            "(sample_id BIGINT, present BOOLEAN);"
        )
        cursor.execute("TRUNCATE _present_update;")
        cursor.copy_from(buf, "_present_update", columns=("sample_id", "present"))
        cursor.execute("CREATE INDEX ON _present_update (sample_id);")
        cursor.execute(
            f"UPDATE {table} t SET present = u.present "
            f"FROM _present_update u WHERE t.sample_id = u.sample_id;"
        )
        print(f"table {table} update complete")
    else:
        print(f"skipping table {table}")


def process_table_worker(table):
    process_table(table)


def main():
    conn, cursor = db_connect()
    tables = get_table_list(conn, cursor)
    db_close(conn, cursor)

    workers = min(8, len(tables))
    with Pool(processes=workers) as pool:
        pool.map(process_table_worker, tables)


if __name__ == "__main__":
    main()
