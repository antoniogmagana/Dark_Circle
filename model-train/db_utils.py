import psycopg2
from psycopg2 import sql
import re
from config import DB_CONN_PARAMS


def sanitize_name(name, max_length=25):
    clean_name = str(name).lower()
    clean_name = re.sub(r"[^a-z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name)
    clean_name = clean_name.strip("_")
    if clean_name and clean_name[0].isdigit():
        clean_name = f"v_{clean_name}"
    clean_name = clean_name[:max_length]
    return clean_name or "unknown_entity"


def db_connect():
    try:
        conn = psycopg2.connect(**DB_CONN_PARAMS)
        conn.autocommit = True
        cursor = conn.cursor()
        print("Connected to PostgreSQL successfully.")
        return conn, cursor
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise


def db_close(conn, cursor):
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------
# 1. Time bounds WITH optional run_id filtering
# ---------------------------------------------------------------------
def get_time_bounds(cursor, table_name, run_id=None):
    """
    Returns (min_timestamp, max_timestamp) for a table.
    If run_id is provided, restricts to that run.
    """

    if run_id is None:
        query = sql.SQL("SELECT MIN(time_stamp), MAX(time_stamp) FROM {table}").format(
            table=sql.Identifier(table_name)
        )
    else:
        query = sql.SQL(
            "SELECT MIN(time_stamp), MAX(time_stamp) "
            "FROM {table} WHERE run_id = {run_id}"
        ).format(
            table=sql.Identifier(table_name),
            run_id=sql.Literal(run_id),
        )

    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] or 0.0, result[1] or 0.0
    except Exception as e:
        print(f"Error fetching bounds for {table_name}: {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------
# 2. Fetch N samples starting at a timestamp WITH optional run_id
# ---------------------------------------------------------------------
def fetch_sensor_batch(cursor, table_name, sample_count, start_time, run_id=None):
    """
    Fetches `sample_count` rows starting at `start_time`.
    Uses the B-tree index on time_stamp (and run_id if present).
    Handles audio, seismic, and 3-axis accelerometer tables.
    """

    # Determine which columns to fetch based on TRAIN_SENSORS
    if "_accel_" in table_name:
        target_cols = sql.SQL("accel_x_ew, accel_y_ns, accel_z_ud")
    else:
        target_cols = sql.SQL("amplitude")

    # WHERE clause depends on whether run_id is used
    if run_id is None:
        where_clause = sql.SQL("time_stamp >= {start_time}").format(
            start_time=sql.Literal(start_time)
        )
    else:
        where_clause = sql.SQL(
            "time_stamp >= {start_time} AND run_id = {run_id}"
        ).format(
            start_time=sql.Literal(start_time),
            run_id=sql.Literal(run_id),
        )

    query = sql.SQL(
        """
        SELECT {columns}
        FROM {table}
        WHERE {where_clause}
        ORDER BY time_stamp ASC
        LIMIT {limit}
        """
    ).format(
        columns=target_cols,
        table=sql.Identifier(table_name),
        where_clause=where_clause,
        limit=sql.Literal(sample_count),
    )

    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []
