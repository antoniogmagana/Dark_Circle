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

def sanitize_name(name, max_length=25):
    """Refines a string to be a safe, clean PostgreSQL identifier."""
    clean_name = str(name).lower()
    clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)
    clean_name = re.sub(r'_+', '_', clean_name)
    clean_name = clean_name.strip('_')
    
    if clean_name and clean_name[0].isdigit():
        clean_name = f"v_{clean_name}"
        
    clean_name = clean_name[:max_length]
    
    if not clean_name:
        clean_name = "unknown_entity"
        
    return clean_name

def create_dynamic_table(conn, cursor, base_name, vehicle, sensor, schema_type):
    """
    Creates a unique table for a specific vehicle and sensor.
    schema_type: 'standard', 'triaxial', 'm3nvc_audio', or 'm3nvc_seismic'
    """
    v_clean = sanitize_name(vehicle)
    s_clean = sanitize_name(sensor)
    table_name = f"{base_name}_{v_clean}_{s_clean}"
    
    schemas = {
        "standard": [
            "sample_id BIGSERIAL PRIMARY KEY",
            "time_stamp REAL NOT NULL",
            "amplitude REAL NOT NULL"
        ],
        "triaxial": [
            "sample_id BIGSERIAL PRIMARY KEY",
            "time_stamp REAL NOT NULL",
            "accel_x_ew REAL",
            "accel_y_ns REAL",
            "accel_z_ud REAL"
        ],
        "m3nvc_audio": [
            "sample_id BIGSERIAL PRIMARY KEY",
            "scene_id INTEGER",
            "run_id INTEGER",
            "time_stamp REAL NOT NULL",
            "amplitude REAL"
        ],
        "m3nvc_seismic": [
            "sample_id BIGSERIAL PRIMARY KEY",
            "scene_id INTEGER",
            "run_id INTEGER",
            "time_stamp REAL NOT NULL",
            "channel VARCHAR(10)",
            "amplitude REAL"
        ]
    }
    
    cols = schemas.get(schema_type)
    if not cols:
        print(f"Unknown schema type: {schema_type}")
        return None
        
    command = f"CREATE TABLE IF NOT EXISTS {table_name} (\n\t{',\n'.join(cols)});"
    
    try:
        cursor.execute(command)
        return table_name
    except Exception as e:
        print(f"Failed to create dynamic table {table_name}: {e}")
        return None

def get_scene_id(conn, cursor, scene_name):
    """Maintains a small registry table for M3N-VC scenes to save space in data tables."""
    check_query = "SELECT scene_id FROM scene_ids WHERE scene = %s"
    try:
        cursor.execute(check_query, (scene_name,))
        result = cursor.fetchone()
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking scene_id for {scene_name}: {e}")
        return None

    if result:
        return result[0]
    else:
        sql_insert_query = "INSERT INTO scene_ids (scene) VALUES (%s) RETURNING scene_id"
        cursor.execute(sql_insert_query, (scene_name,))
        return cursor.fetchone()[0]

def copy_to_postgres(conn, cursor, df, table_name, db_columns):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="0")
    buffer.seek(0)
    try:
        cursor.copy_from(buffer, table_name, sep=",", columns=db_columns)
    except Exception as e:
        print(f"Error copying to {table_name}: {e}")
        # Rollback is critical here so a failed copy doesn't break subsequent inserts
        conn.rollback() 


# ==========================================
# Original Loaders (MOD_vehicle / CSV)
# ==========================================

def load_data(conn, cursor, data_set, root_dir, base_table_name, file_name, sep, data_cols, data_col_idx, sample_period):
    files = list(root_dir.rglob(file_name))
    print(f"Found {len(files)} files for {base_table_name}. Starting COPY process...")

    for file_path in files:
        parts = file_path.parts
        try:
            MOD_idx = parts.index(data_set)
            vehicle = parts[MOD_idx + 1]
            sensor = parts[MOD_idx + 2]
        except (ValueError, IndexError):
            continue

        table_name = create_dynamic_table(conn, cursor, base_table_name, vehicle, sensor, "standard")
        if not table_name:
            continue
            
        print(f"Streaming {vehicle} sensor {sensor} into {table_name}...")
        samples_processed = 0

        for chunk in pd.read_csv(
            file_path, sep=sep, header=None, names=list(data_cols.keys()),
            usecols=data_col_idx, dtype=data_cols, chunksize=variables.CHUNK_SIZE
        ):
            chunk["time_stamp"] = (pd.Series(range(len(chunk))) + samples_processed) * sample_period
            
            final_chunk = chunk[["time_stamp", "amplitude"]]
            copy_to_postgres(conn, cursor, final_chunk, table_name, ("time_stamp", "amplitude"))
            
            samples_processed += len(chunk)


def load_tri_axial_data(conn, cursor, data_set, root_dir, base_table_name, file_map, sample_period):
    primary_col = list(file_map.keys())[0]
    primary_file = file_map[primary_col]

    files = list(root_dir.rglob(primary_file))
    print(f"Found {len(files)} potential tri-axial sets for {base_table_name}. Processing...")

    for file_path in files:
        parent_dir = file_path.parent
        parts = file_path.parts
        
        try:
            MOD_idx = parts.index(data_set)
            vehicle = parts[MOD_idx + 1]
            sensor = parts[MOD_idx + 2]
        except (ValueError, IndexError):
            continue

        dfs = []
        valid_set = True
        
        for db_col, fname in file_map.items():
            target_file = parent_dir / fname
            if not target_file.exists():
                valid_set = False
                break
            dfs.append(pd.read_csv(target_file, header=None, names=[db_col]))

        if not valid_set:
            continue

        table_name = create_dynamic_table(conn, cursor, base_table_name, vehicle, sensor, "triaxial")
        if not table_name:
            continue

        combined_df = pd.concat(dfs, axis=1)
        combined_df["time_stamp"] = np.arange(len(combined_df)) * sample_period
        
        db_cols = ["time_stamp"] + list(file_map.keys())
        final_df = combined_df[db_cols]

        copy_to_postgres(conn, cursor, final_df, table_name, tuple(db_cols))
        print(f"   - Inserted {len(final_df)} rows into {table_name}")


# ==========================================
# New Loader (M3N-VC / Parquet)
# ==========================================

def load_m3nvc_dataset(conn, cursor, root_path):
    root = Path(root_path).expanduser()
    if not root.exists():
        print(f"Path {root} does not exist. Skipping M3N-VC.")
        return

    scenes = [d for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(scenes)} scenes in {root_path}")

    for scene in scenes:
        print(f"Processing Scene: {scene.name}")
        sc_id = get_scene_id(conn, cursor, scene.name)
        
        meta_path = scene / "run_ids.parquet"
        run_map = {}

        if meta_path.exists():
            try:
                meta_df = pd.read_parquet(meta_path)
                meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]

                def clean_label(val):
                    if isinstance(val, (list, np.ndarray)):
                        return "+".join(str(v) for v in val)
                    return val

                if "label" in meta_df.columns:
                    meta_df["label"] = meta_df["label"].apply(clean_label)

                meta_df = meta_df.drop_duplicates(subset=["run_id"])
                run_map = meta_df.set_index("run_id")["label"].to_dict()

            except Exception as e:
                print(f"  - Failed to load metadata properly: {e}")

        for p_file in scene.glob("*.parquet"):
            fname = p_file.name

            if fname in ["run_ids.parquet", "sensor_location.parquet"] or "gps" in fname or "dis" in fname:
                continue

            match = re.match(r"run(\d+)_rs(\d+)_([a-z]{3})\.parquet", fname)
            if not match:
                continue

            run_num = int(match.group(1))
            sensor_node = f"rs{match.group(2)}"
            modality = match.group(3)

            if modality == "mic":
                base_table_name = "m3nvc_audio"
                schema_type = "m3nvc_audio"
                sample_period = variables.ACOUSTIC_PR2
                db_cols = ("scene_id", "run_id", "time_stamp", "amplitude")
            elif modality == "geo":
                base_table_name = "m3nvc_seismic"
                schema_type = "m3nvc_seismic"
                sample_period = variables.SEISMIC_PR2
                db_cols = ("scene_id", "run_id", "time_stamp", "channel", "amplitude")
            else:
                continue

            vehicle_label = run_map.get(run_num)
            if run_num == 8 or run_num == 9:
                vehicle_label = "background"

            if not vehicle_label:
                continue

            table_name = create_dynamic_table(conn, cursor, base_table_name, vehicle_label, sensor_node, schema_type)
            if not table_name:
                continue

            try:
                df = pd.read_parquet(p_file)
                if df.empty:
                    continue

                df = df.sort_values("timestamp")
                df["time_stamp"] = np.arange(len(df)) * sample_period
                df = df.rename(columns={"samples": "amplitude"})

                df["scene_id"] = sc_id
                df["run_id"] = run_num

                if modality == "mic":
                    final_df = df[["scene_id", "run_id", "time_stamp", "amplitude"]]
                else:
                    if "channel" not in df.columns:
                        df["channel"] = "UD"
                    final_df = df[["scene_id", "run_id", "time_stamp", "channel", "amplitude"]]

                copy_to_postgres(conn, cursor, final_df, table_name, db_cols)

            except Exception as e:
                print(f"  - Error processing {fname}: {e}")
                
    print("Finished loading M3N-VC dataset.")


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":

    conn, cursor = db_connect()

    # 1. Generate Metadata Tables (Only scene_ids is needed now)
    try:
        cursor.execute("CREATE TABLE IF NOT EXISTS scene_ids (scene_id SERIAL PRIMARY KEY, scene VARCHAR(50));")
        print("scene_ids registry table confirmed.")
    except Exception as e:
        print(f"Failed to create scene_ids table: {e}")

    # 2. Load Original Dataset
    load_data(
        conn, cursor, "MOD_vehicle", Path(variables.MOD_path).expanduser(),
        "audio_16k", "aud16000.csv", ",", {"amplitude": "float32"}, [0], variables.ACOUSTIC_PR
    )

    load_data(
        conn, cursor, "MOD_vehicle", Path(variables.MOD_path).expanduser(),
        "audio_raw", "aud.csv", ",", {"amplitude": "float32", "raw": "float32"}, [0], variables.ACOUSTIC_PR
    )

    load_data(
        conn, cursor, "MOD_vehicle", Path(variables.MOD_path).expanduser(),
        "seismic", "ehz.csv", " ", {"amplitude": "float32"}, [0], variables.SEISMIC_PR
    )

    accel_mapping = {
        "accel_x_ew": "ene.csv",
        "accel_y_ns": "enn.csv",
        "accel_z_ud": "enz.csv",
    }
    
    load_tri_axial_data(
        conn, cursor, "MOD_vehicle", Path(variables.MOD_path).expanduser(),
        "accel", accel_mapping, variables.SEISMIC_PR
    )

    # 3. Load M3N-VC Dataset
    print("\n--- Starting M3N-VC Import ---")
    load_m3nvc_dataset(conn, cursor, variables.M3NVC_path)

    db_close(conn, cursor)