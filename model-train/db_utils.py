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


def get_time_bounds(cursor, table_name):
    """
    Fetches the actual start and end time of a specific table.
    Crucial for M3NVC background runs which don't start at T=0.
    """
    query = sql.SQL("SELECT MIN(time_stamp), MAX(time_stamp) FROM {table}").format(
        table=sql.Identifier(table_name)
    )
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] or 0.0, result[1] or 0.0
    except Exception as e:
        print(f"Error fetching bounds for {table_name}: {e}")
        return 0.0, 0.0


def fetch_sensor_batch(cursor, table_name, sample_rate, start_time):
    """
    Fetches exactly 1 second of data starting from a precise timestamp.
    Uses the B-Tree index on time_stamp for lightning-fast retrieval.
    """
    if "_accel_" in table_name:
        target_cols = "accel_x_ew, accel_y_ns, accel_z_ud"
    else:
        target_cols = "amplitude"

    # We fetch enough rows to satisfy 1 second of data based on the sample rate.
    # WHERE time_stamp >= start_time leverages the index instantly.
    query = sql.SQL(
        """
        SELECT {columns} 
        FROM {table} 
        WHERE time_stamp >= {start_time} 
        ORDER BY time_stamp ASC 
        LIMIT {limit}
        """
    ).format(
        columns=sql.SQL(target_cols),
        table=sql.Identifier(table_name),
        start_time=sql.Literal(start_time),
        limit=sql.Literal(sample_rate),  # 1 second = sample_rate rows
    )

    try:
        cursor.execute(query)
        return cursor.fetchall()
    except (Exception, psycopg2.Error) as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []
