import os
import io
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
import variables


def db_connect():
    try:
        conn = psycopg2.connect(**variables.conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"Failed to connect: {e}")
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
    # 1. Check if it exists using a parameterized query (%s)
    check_query = """
        SELECT EXISTS (
            SELECT 1
            FROM vehicle_ids
            WHERE vehicle = %s
        );
    """

    try:
        # Pass the parameter as a tuple (vehicle_name,)
        cursor.execute(check_query, (vehicle_name,))
        exists = cursor.fetchone()[0]

    except (Exception, psycopg2.Error) as e:
        print(f"Error checking vehicle_id:", e)
        return None  # Return early or handle the error appropriately

    # 2. Retrieve or Insert
    if exists:
        sql_select_query = "SELECT vehicle_id FROM vehicle_ids WHERE vehicle = %s"
        cursor.execute(sql_select_query, (vehicle_name,))
        result = cursor.fetchone()[0]  # Make sure to grab the ID [0]
    else:
        # Assuming vehicle_id is SERIAL/AUTO INCREMENT, we use RETURNING to get the new ID
        sql_insert_query = (
            "INSERT INTO vehicle_ids (vehicle) VALUES (%s) RETURNING vehicle_id"
        )
        cursor.execute(sql_insert_query, (vehicle_name,))
        conn.commit()  # Commit the insert
        result = cursor.fetchone()[0]

    return result


def get_sensor_id(conn, cursor, sensor_name):
    # 1. Parameterized Check
    # We use %s to let the driver handle quoting and safety
    check_query = """
    SELECT EXISTS (
        SELECT 1
        FROM sensor_ids
        WHERE sensor = %s
    );
    """
    try:
        cursor.execute(check_query, (sensor_name,))
        exists = cursor.fetchone()[0]
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking sensor_id:", e)
        # Return None to avoid the UnboundLocalError later
        return None

    # 2. Retrieve or Insert
    if exists:
        sql_select_query = "SELECT sensor_id FROM sensor_ids WHERE sensor = %s"
        cursor.execute(sql_select_query, (sensor_name,))
        # Standardize return: get the integer ID, not the tuple
        result = cursor.fetchone()[0]
    else:
        # Added RETURNING sensor_id to ensure we get the ID back after insert
        sql_insert_query = (
            "INSERT INTO sensor_ids (sensor) VALUES (%s) RETURNING sensor_id"
        )
        cursor.execute(sql_insert_query, (sensor_name,))
        conn.commit()
        result = cursor.fetchone()[0]

    return result


def copy_to_postgres(conn, cursor, df, table_name, value_column):
    """Efficiently streams a dataframe to Postgres using COPY."""
    buffer = io.StringIO()

    # na_rep='0' ensures if a NaN slips in, it doesn't break the COPY command
    # quoting=None and doublequote=False keep the CSV "lean" for the COPY command
    df.to_csv(buffer, index=False, header=False, na_rep="0")

    buffer.seek(0)
    try:
        cursor.copy_from(
            buffer,
            table_name,
            sep=",",
            columns=("vehicle_id", "sensor_id", "time_stamp", value_column),
        )
    except Exception as e:
        # Debugging: if it fails, show us the first few lines of the buffer
        buffer.seek(0)
        print(f"First line of failed buffer: {buffer.readline()}")
        raise e


# load data from storage to database
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
    print(f"found {len(files)} files. Starting COPY process...")

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

            copy_to_postgres(conn, cursor, chunk, table_name, "amplitude")

            samples_processed += len(chunk)
            print(f"   - Bulk inserted {samples_processed} samples...")


def load_m3nvc_audio(conn, cursor):
    pass


def load_m3nvc_seismic(conn, cursor):
    pass


# main script
if __name__ == "__main__":

    # open database connection
    conn, cursor = db_connect()

    # generate vehicle index table
    generate_table(
        conn,
        cursor,
        "vehicle_ids",
        ["vehicle_id SERIAL PRIMARY KEY", "vehicle VARCHAR(50)"],
    )

    # generate sensor index table
    generate_table(
        conn,
        cursor,
        "sensor_ids",
        ["sensor_id SERIAL PRIMARY KEY", "sensor VARCHAR(3)"],
    )

    # generate audio table
    generate_table(
        conn,
        cursor,
        "audio_data",
        [
            "sample_id SERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "time_stamp REAL NOT NULL",
            "amplitude REAL NOT NULL",
        ],
    )

    # generate seismic table
    generate_table(
        conn,
        cursor,
        "seismic_data",
        [
            "sample_id SERIAL PRIMARY KEY",
            "vehicle_id INTEGER NOT NULL",
            "sensor_id INTEGER NOT NULL",
            "time_stamp REAL NOT NULL",
            "amplitude REAL NOT NULL",
        ],
    )

    # load database
    # load_iobt_audio
    load_data(
        conn,
        cursor,
        "MOD_vehicle",
        Path(variables.MOD_path),
        "audio_data",
        "aud16000.csv",
        ",",
        {"amplitude": "float32"},
        [0],
        variables.ACOUSTIC_PR,
    )
    # load focal audio
    load_data(
        conn,
        cursor,
        "MOD_vehicle",
        Path(variables.MOD_path),
        "audio_data",
        "aud.csv",
        ",",
        {"amplitude": "float32", "raw": "float32"},
        [0],
        variables.ACOUSTIC_PR,
    )
    # load iobt and focal geophone seismic
    load_data(
        conn,
        cursor,
        "MOD_vehicle",
        Path(variables.MOD_path),
        "seismic_data",
        "ehz.csv",
        " ",
        {"amplitude": "float32"},
        [0],
        variables.SEISMIC_PR,
    )

    # close database connection
    db_close(conn, cursor)
