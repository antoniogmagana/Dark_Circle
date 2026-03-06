import psycopg2
from psycopg2 import sql
import re
from config import DB_CONN_PARAMS


def sanitize_name(name, max_length=25):
    """Refines a string to be a safe, clean PostgreSQL identifier."""
    clean_name = str(name).lower()
    clean_name = re.sub(r"[^a-z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name)
    clean_name = clean_name.strip("_")
    if clean_name and clean_name[0].isdigit():
        clean_name = f"v_{clean_name}"
    clean_name = clean_name[:max_length]
    if not clean_name:
        clean_name = "unknown_entity"
    return clean_name


def db_connect():
    try:
        conn = psycopg2.connect(**DB_CONN_PARAMS)
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


def fetch_sensor_batch(cursor, table_name, limit, offset):
    """
    Fetches a precise window of data.
    Infers schema (amplitude vs tri-axial) directly from the table name.
    """
    if "_accel_" in table_name:
        target_cols = "accel_x_ew, accel_y_ns, accel_z_ud"
    else:
        target_cols = "amplitude"

    query = sql.SQL(
        "SELECT {columns} FROM {table} ORDER BY time_stamp ASC LIMIT {limit} OFFSET {offset}"
    ).format(
        columns=sql.SQL(target_cols),
        table=sql.Identifier(table_name),
        limit=sql.Literal(limit),
        offset=sql.Literal(offset),
    )

    try:
        cursor.execute(query)
        return cursor.fetchall()

    except (Exception, psycopg2.Error) as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []
