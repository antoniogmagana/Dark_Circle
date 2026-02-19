import os
import io
import re
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
import variables

# ==========================================
# Database Helper Functions
# ==========================================


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


def generate_table(conn, cursor, name, cols):
    command = f"""
    CREATE TABLE IF NOT EXISTS {name} (\n\t{',\n'.join(cols)});
    """
    try:
        cursor.execute(command)
        conn.commit()
        print(f"table {name} created.")
    except Exception as e:
        print(f"Failed to create table {name}: {e}")
        return


def get_vehicle_id(conn, cursor, vehicle_name):
    # 1. Check if it exists
    check_query = "SELECT vehicle_id FROM vehicle_ids WHERE vehicle = %s"
    try:
        cursor.execute(check_query, (vehicle_name,))
        result = cursor.fetchone()
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking vehicle_id for {vehicle_name}: {e}")
        return None

    # 2. Retrieve or Insert
    if result:
        return result[0]
    else:
        sql_insert_query = (
            "INSERT INTO vehicle_ids (vehicle) VALUES (%s) RETURNING vehicle_id"
        )
        cursor.execute(sql_insert_query, (vehicle_name,))
        conn.commit()
        return cursor.fetchone()[0]


def get_sensor_id(conn, cursor, sensor_name):
    # 1. Check if it exists
    check_query = "SELECT sensor_id FROM sensor_ids WHERE sensor = %s"
    try:
        cursor.execute(check_query, (sensor_name,))
        result = cursor.fetchone()
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking sensor_id for {sensor_name}: {e}")
        return None

    # 2. Retrieve or Insert
    if result:
        return result[0]
    else:
        sql_insert_query = (
            "INSERT INTO sensor_ids (sensor) VALUES (%s) RETURNING sensor_id"
        )
        cursor.execute(sql_insert_query, (sensor_name,))
        conn.commit()
        return cursor.fetchone()[0]


def copy_to_postgres(conn, cursor, df, table_name, db_columns):
    """
    Efficiently streams a dataframe to Postgres using COPY.
    db_columns: tuple of column names in the DB to map the DF columns to.
    """
    buffer = io.StringIO()

    # Write to buffer (index=False, header=False)
    df.to_csv(buffer, index=False, header=False, na_rep="0")
    buffer.seek(0)

    try:
        cursor.copy_from(buffer, table_name, sep=",", columns=db_columns)
    except Exception as e:
        print(f"Error copying to {table_name}: {e}")
        raise e


# ==========================================
# Original Loaders (MOD_vehicle / CSV)
# ==========================================


def load_data(
    conn,
    cursor,
    data_set,
    root_dir,
    table_name,
    file_name,
    sep,
    data_cols,
    data_col_idx,
    sample_period,
):
    files = list(root_dir.rglob(file_name))
    print(f"found {len(files)} files for {table_name}. Starting COPY process...")

    for file_path in files:
        parts = file_path.parts
        try:
            MOD_idx = parts.index(data_set)
            vehicle = parts[MOD_idx + 1]
            sensor = parts[MOD_idx + 2]
        except (ValueError, IndexError):
            continue

        print(f"Streaming {vehicle} sensor {sensor}...")
        samples_processed = 0

        # Read CSV in chunks
        for chunk in pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            names=list(data_cols.keys()),
            usecols=data_col_idx,
            dtype=data_cols,
            chunksize=variables.CHUNK_SIZE,
        ):

            chunk_idx = pd.Series(range(len(chunk))) + samples_processed
            chunk["time_stamp"] = chunk_idx * sample_period
            chunk["vehicle_id"] = get_vehicle_id(conn, cursor, vehicle)
            chunk["sensor_id"] = get_sensor_id(conn, cursor, sensor)

            chunk = chunk[["vehicle_id", "sensor_id", "time_stamp", "amplitude"]]

            copy_to_postgres(
                conn,
                cursor,
                chunk,
                table_name,
                ("vehicle_id", "sensor_id", "time_stamp", "amplitude"),
            )

            samples_processed += len(chunk)


def load_tri_axial_data(
    conn,
    cursor,
    data_set,
    root_dir,
    table_name,
    file_map,  # Dictionary mapping DB column -> filename
    sample_period,
):
    """
    Scans for the first file in file_map, checks if siblings exist,
    merges them, and uploads.
    """
    primary_col = list(file_map.keys())[0]
    primary_file = file_map[primary_col]

    files = list(root_dir.rglob(primary_file))
    print(f"Found {len(files)} potential accelerometer sets. Processing...")

    for file_path in files:
        parent_dir = file_path.parent

        # 1. Parse Path for Metadata (Vehicle/Sensor)
        parts = file_path.parts
        try:
            MOD_idx = parts.index(data_set)
            vehicle = parts[MOD_idx + 1]
            sensor = parts[MOD_idx + 2]
        except (ValueError, IndexError):
            print(f"Skipping {file_path}, could not parse path structure.")
            continue

        # 2. Check if all 3 files exist
        dfs = []
        valid_set = True

        # We need these columns for the DB
        db_cols = ["vehicle_id", "sensor_id", "time_stamp"] + list(file_map.keys())

        for db_col, fname in file_map.items():
            target_file = parent_dir / fname
            if not target_file.exists():
                print(f"Missing sibling file {fname} in {parent_dir}. Skipping set.")
                valid_set = False
                break

            # Read CSV
            temp_df = pd.read_csv(target_file, header=None, names=[db_col])
            dfs.append(temp_df)

        if not valid_set:
            continue

        # 3. Merge Dataframes Side-by-Side
        try:
            combined_df = pd.concat(dfs, axis=1)
        except Exception as e:
            print(f"Error merging files in {parent_dir}: {e}")
            continue

        # 4. Add Metadata and Time
        chunk_idx = pd.Series(range(len(combined_df)))
        combined_df["time_stamp"] = chunk_idx * sample_period

        # Get IDs
        v_id = get_vehicle_id(conn, cursor, vehicle)
        s_id = get_sensor_id(conn, cursor, sensor)

        combined_df["vehicle_id"] = v_id
        combined_df["sensor_id"] = s_id

        # 5. Reorder to match DB schema exactly
        final_df = combined_df[db_cols]

        # 6. Upload
        copy_to_postgres(conn, cursor, final_df, table_name, tuple(db_cols))
        print(f"   - Inserted {len(final_df)} rows for {vehicle} {sensor}")


# ==========================================
# New Loader (M3N-VC / Parquet) - RELATIVE TIME
# ==========================================


def load_m3nvc_dataset(conn, cursor, root_path):
    """
    Loads M3N-VC parquet files.
    - Handles 'i22' list-based labels by joining them (e.g., "gle350+mustang").
    - Manually assigns "background" label to Runs 8 and 9.
    - Skips empty files and handles duplicate metadata columns.
    """
    root = Path(root_path).expanduser()
    if not root.exists():
        print(f"Path {root} does not exist. Skipping M3N-VC.")
        return

    # Iterate over Scene folders (h08, s31, etc.)
    scenes = [d for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(scenes)} scenes in {root_path}")

    for scene in scenes:
        print(f"Processing Scene: {scene.name}")

        # 1. Load Metadata (Run IDs)
        meta_path = scene / "run_ids.parquet"
        run_map = {}

        if meta_path.exists():
            try:
                meta_df = pd.read_parquet(meta_path)

                # --- FIX 1: Handle Duplicate Columns (e.g. two 'label' columns) ---
                meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]

                # --- FIX 2: Handle Lists in Label Column (for scene i22) ---
                def clean_label(val):
                    if isinstance(val, (list, np.ndarray)):
                        return "+".join(str(v) for v in val)
                    return val

                if "label" in meta_df.columns:
                    meta_df["label"] = meta_df["label"].apply(clean_label)

                # Create map: int(run_id) -> str(label)
                # Drop duplicates to prevent errors if run_id appears twice
                meta_df = meta_df.drop_duplicates(subset=["run_id"])
                run_map = meta_df.set_index("run_id")["label"].to_dict()

            except Exception as e:
                print(f"  - Failed to load metadata properly: {e}")
                # We continue, because we might still be able to load runs 8/9 manually
        else:
            print(
                f"  - Missing run_ids.parquet in {scene.name}. Only manual runs will load."
            )

        # 2. Process Sensor Files
        for p_file in scene.glob("*.parquet"):
            fname = p_file.name

            # Skip metadata/gps files
            if (
                fname in ["run_ids.parquet", "sensor_location.parquet"]
                or "gps" in fname
                or "dis" in fname
            ):
                continue

            # Parse Filename: run<ID>_rs<ID>_<type>.parquet
            match = re.match(r"run(\d+)_rs(\d+)_([a-z]{3})\.parquet", fname)
            if not match:
                continue

            run_num = int(match.group(1))
            sensor_node = f"rs{match.group(2)}"
            modality = match.group(3)

            # Identify Target Table
            if modality == "mic":
                table_name = "m3nvc_audio"
                db_cols = (
                    "vehicle_id",
                    "sensor_id",
                    "run_id",
                    "time_stamp",
                    "amplitude",
                )
            elif modality == "geo":
                table_name = "m3nvc_seismic"
                db_cols = (
                    "vehicle_id",
                    "sensor_id",
                    "run_id",
                    "time_stamp",
                    "channel",
                    "amplitude",
                )
            else:
                continue

            # --- FIX 3: Get Label or Apply Manual Override ---
            vehicle_label = run_map.get(run_num)

            # Manually handle known background runs missing from metadata
            if run_num == 8:
                vehicle_label = "background"
            elif run_num == 9:
                vehicle_label = "background"

            # If still no label (unknown run), skip it
            if not vehicle_label:
                # print(f"  - Warning: Unknown run {run_num} in {fname}")
                continue

            # Get DB IDs
            v_id = get_vehicle_id(conn, cursor, str(vehicle_label))
            s_id = get_sensor_id(conn, cursor, sensor_node)

            # Load Data
            try:
                df = pd.read_parquet(p_file)

                # --- FIX 4: Check for Empty Files ---
                if df.empty:
                    print(f"  - Warning: File {fname} is empty. Skipping.")
                    continue

                # --- TIME CALCULATION: ABSOLUTE TO RELATIVE ---
                # This uses the file's OWN timestamps, so missing metadata start/stop doesn't matter.
                df = df.sort_values("timestamp")
                start_time = df["timestamp"].iloc[0]
                df["timestamp"] = df["timestamp"] - start_time

                # Rename columns
                df = df.rename(
                    columns={"timestamp": "time_stamp", "samples": "amplitude"}
                )

                # Add Foreign Keys
                df["vehicle_id"] = v_id
                df["sensor_id"] = s_id
                df["run_id"] = run_num

                # Select Columns
                if modality == "mic":
                    final_df = df[
                        ["vehicle_id", "sensor_id", "run_id", "time_stamp", "amplitude"]
                    ]
                else:
                    if "channel" not in df.columns:
                        df["channel"] = "UD"
                    final_df = df[
                        [
                            "vehicle_id",
                            "sensor_id",
                            "run_id",
                            "time_stamp",
                            "channel",
                            "amplitude",
                        ]
                    ]

                copy_to_postgres(conn, cursor, final_df, table_name, db_cols)

            except Exception as e:
                print(f"  - Error processing {fname}: {e}")
    print("Finished loading M3N-VC dataset.")


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":

    conn, cursor = db_connect()

    # 1. Generate ID Tables
    generate_table(
        conn,
        cursor,
        "vehicle_ids",
        ["vehicle_id SERIAL PRIMARY KEY", "vehicle VARCHAR(50)"],
    )
    generate_table(
        conn,
        cursor,
        "sensor_ids",
        ["sensor_id SERIAL PRIMARY KEY", "sensor VARCHAR(10)"],
    )

    # 2. Generate Original Tables
    generate_table(
        conn,
        cursor,
        "audio_data",
        [
            "sample_id BIGSERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "time_stamp REAL NOT NULL",
            "amplitude REAL NOT NULL",
        ],
    )

    generate_table(
        conn,
        cursor,
        "seismic_data",
        [
            "sample_id BIGSERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "time_stamp REAL NOT NULL",
            "amplitude REAL NOT NULL",
        ],
    )

    generate_table(
        conn,
        cursor,
        "accelerometer_data",
        [
            "sample_id BIGSERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "time_stamp REAL NOT NULL",
            "accel_x_ew REAL",
            "accel_y_ns REAL",
            "accel_z_ud REAL",
        ],
    )

    # 3. Generate M3N-VC Tables (Using REAL for relative time)
    generate_table(
        conn,
        cursor,
        "m3nvc_audio",
        [
            "sample_id BIGSERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "run_id INTEGER",
            "time_stamp REAL NOT NULL",  # Changed to REAL for relative time
            "amplitude REAL",
        ],
    )

    generate_table(
        conn,
        cursor,
        "m3nvc_seismic",
        [
            "sample_id BIGSERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "run_id INTEGER",
            "time_stamp REAL NOT NULL",  # Changed to REAL for relative time
            "channel VARCHAR(10)",
            "amplitude REAL",
        ],
    )

    # # 4. Load Original Dataset
    # load_data(
    #     conn,
    #     cursor,
    #     "MOD_vehicle",
    #     Path(variables.MOD_path).expanduser(),
    #     "audio_data",
    #     "aud16000.csv",
    #     ",",
    #     {"amplitude": "float32"},
    #     [0],
    #     variables.ACOUSTIC_PR,
    # )

    # load_data(
    #     conn,
    #     cursor,
    #     "MOD_vehicle",
    #     Path(variables.MOD_path).expanduser(),
    #     "audio_data",
    #     "aud.csv",
    #     ",",
    #     {"amplitude": "float32", "raw": "float32"},
    #     [0],
    #     variables.ACOUSTIC_PR,
    # )

    # load_data(
    #     conn,
    #     cursor,
    #     "MOD_vehicle",
    #     Path(variables.MOD_path).expanduser(),
    #     "seismic_data",
    #     "ehz.csv",
    #     " ",
    #     {"amplitude": "float32"},
    #     [0],
    #     variables.SEISMIC_PR,
    # )

    # accel_mapping = {
    #     "accel_x_ew": "ene.csv",
    #     "accel_y_ns": "enn.csv",
    #     "accel_z_ud": "enz.csv",
    # }
    # load_tri_axial_data(
    #     conn,
    #     cursor,
    #     "MOD_vehicle",
    #     Path(variables.MOD_path).expanduser(),
    #     "accelerometer_data",
    #     accel_mapping,
    #     variables.SEISMIC_PR,
    # )

    # 5. Load M3N-VC Dataset
    print("\n--- Starting M3N-VC Import ---")
    load_m3nvc_dataset(conn, cursor, variables.M3NVC_path)

    db_close(conn, cursor)
