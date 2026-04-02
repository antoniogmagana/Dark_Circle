import argparse
import sys
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
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

NOISE_FACTOR = 10

SAMPLE_WINDOW = 1.0
EXTENDED_WINDOW = 3.0

MIN_PRESENCE = 0.5
THRESHOLD_DB = 3.0

ROOT = str(".")


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


def label_windows(amplitudes, sample_rate):
    window_size = int(SAMPLE_WINDOW * sample_rate)
    num_windows = len(amplitudes) // window_size
    if num_windows == 0:
        return np.array([], dtype=bool)

    windows = amplitudes[: num_windows * window_size].reshape(num_windows, window_size)

    # Build 3-second context: [prev | current | next], clamped at boundaries
    prev = np.concatenate([windows[:1], windows[:-1]], axis=0)
    next_ = np.concatenate([windows[1:], windows[-1:]], axis=0)
    ctx = np.concatenate([prev, windows, next_], axis=1)

    # Vectorized noise floor: 10th percentile of abs(centered) per context window
    ctx_centered = ctx - ctx.mean(axis=1, keepdims=True)
    noise_floors = np.percentile(np.abs(ctx_centered), NOISE_FACTOR, axis=1)

    # Vectorized vehicle presence: fraction of samples >= 3dB above noise floor
    thresholds = noise_floors * (10 ** (THRESHOLD_DB / 20))
    win_centered = windows - windows.mean(axis=1, keepdims=True)
    present = (
        np.mean(np.abs(win_centered) >= thresholds[:, None], axis=1) >= MIN_PRESENCE
    )

    return present


def process_audio_table(df):
    amplitudes = df["amplitude"].values
    window_size = BASE_RATES["audio"]
    labels = label_windows(amplitudes, BASE_RATES["audio"])
    df["present"] = False
    df.iloc[: len(labels) * window_size, df.columns.get_loc("present")] = np.repeat(
        labels, window_size
    )
    return df


def process_seismic_table(df):
    amplitudes = df["amplitude"].values
    window_size = BASE_RATES["seismic"]
    labels = label_windows(amplitudes, BASE_RATES["seismic"])
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
    labels = label_windows(magnitude.values, BASE_RATES["seismic"])
    df["present"] = False
    df.iloc[: len(labels) * window_size, df.columns.get_loc("present")] = np.repeat(
        labels, window_size
    )
    return df


def process_m3nvc_audio(df):
    df["present"] = False
    window_size = int(SAMPLE_WINDOW * M3NVC_RATES["audio"])
    for (scene_id, run_id), group in df.groupby(["scene_id", "run_id"]):
        amplitudes = group["amplitude"].values
        labels = label_windows(amplitudes, M3NVC_RATES["audio"])
        present_arr = np.repeat(labels, window_size)[: len(group)]
        df.loc[group.index, "present"] = present_arr
    return df


def process_m3nvc_seismic(df):
    df["present"] = False
    window_size = int(SAMPLE_WINDOW * M3NVC_RATES["seismic"])
    for (scene_id, run_id), group in df.groupby(["scene_id", "run_id"]):
        amplitudes = group["amplitude"].values
        labels = label_windows(amplitudes, M3NVC_RATES["seismic"])
        present_arr = np.repeat(labels, window_size)[: len(group)]
        df.loc[group.index, "present"] = present_arr
    return df


def process_table(conn, cursor, table):
    # extract table designations
    table_type = table.split("_")
    # Add default false column for vehicle presence
    query = f"""ALTER TABLE {table} 
    ADD COLUMN IF NOT EXISTS present BOOLEAN NOT NULL DEFAULT TRUE;"""
    cursor.execute(query)
    query = f"""
    UPDATE {table} SET present = TRUE;"""
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
    df = pd.read_sql_query(f"SELECT {cols} FROM {table};", conn)
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
        records = list(zip(df["present"].tolist(), df["sample_id"].tolist()))
        execute_batch(
            cursor,
            f"UPDATE {table} SET present = %s WHERE sample_id = %s",
            records,
            page_size=1000,
        )
        print(f"table {table} update complete")
    else:
        print(f"skipping table {table}")

    # save changes
    conn.commit()


def main():
    # connect to db
    conn, cursor = db_connect()
    # get table list
    tables = get_table_list(conn, cursor)
    for table in tables:
        process_table(conn, cursor, table)


if __name__ == "__main__":
    main()
