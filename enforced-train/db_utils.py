import psycopg2
from psycopg2 import sql
import re

# NOTICE: global 'config' is no longer imported


def sanitize_name(name, max_length=25):
    clean_name = str(name).lower()
    clean_name = re.sub(r"[^a-z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name)
    clean_name = clean_name.strip("_")
    if clean_name and clean_name[0].isdigit():
        clean_name = f"v_{clean_name}"
    clean_name = clean_name[:max_length]
    return clean_name or "unknown_entity"


def db_connect(db_conn_params):
    try:
        # Unpacks the passed dictionary
        conn = psycopg2.connect(**db_conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise


def db_close(conn, cursor):
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------
# 0. Per-second present map from the labeled present column
# ---------------------------------------------------------------------
def get_present_map(cursor, table_name, run_id=None):
    """
    Returns {int(second): bool} for each 1-second labeled block in the table.
    Queries the 'present' column added by sample_parse.py.
    Returns an empty dict on error (default missing keys to True).
    """
    try:
        if run_id is not None:
            query = sql.SQL(
                "SELECT FLOOR(time_stamp)::int AS sec, BOOL_AND(present) "
                "FROM {table} WHERE run_id = {run_id} "
                "GROUP BY sec ORDER BY sec"
            ).format(
                table=sql.Identifier(table_name),
                run_id=sql.Literal(run_id),
            )
        else:
            query = sql.SQL(
                "SELECT FLOOR(time_stamp)::int AS sec, BOOL_AND(present) "
                "FROM {table} GROUP BY sec ORDER BY sec"
            ).format(table=sql.Identifier(table_name))
        cursor.execute(query)
        return {row[0]: row[1] for row in cursor.fetchall()}
    except Exception as e:
        print(f"[WARN] Could not fetch present map for {table_name}: {e}")
        return {}


# ---------------------------------------------------------------------
# 1. Time bounds WITH optional run_id filtering
# ---------------------------------------------------------------------
def get_time_bounds(cursor, table_name, run_id=None):
    """
    Returns (min_timestamp, max_timestamp) for a table.
    If run_id is provided, restricts to that run.
    """

    if run_id is None:
        query = sql.SQL(
            "SELECT MIN(time_stamp), MAX(time_stamp) FROM {table}"
        ).format(
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
# 2. Bulk-fetch an entire table segment from a start time onward
# ---------------------------------------------------------------------
def fetch_table_segment(cursor, table_name, from_time, run_id=None):
    """
    Fetches all rows from `from_time` onward, ordered by time_stamp.
    Used for pre-loading entire table segments into memory at startup.
    """
    if "_accel_" in table_name:
        target_cols = sql.SQL("accel_x_ew, accel_y_ns, accel_z_ud")
    else:
        target_cols = sql.SQL("amplitude")

    if run_id is None:
        query = sql.SQL(
            "SELECT {columns} FROM {table} "
            "WHERE time_stamp >= {from_time} ORDER BY time_stamp ASC"
        ).format(
            columns=target_cols,
            table=sql.Identifier(table_name),
            from_time=sql.Literal(float(from_time)),
        )
    else:
        query = sql.SQL(
            "SELECT {columns} FROM {table} "
            "WHERE time_stamp >= {from_time} AND run_id = {run_id} "
            "ORDER BY time_stamp ASC"
        ).format(
            columns=target_cols,
            table=sql.Identifier(table_name),
            from_time=sql.Literal(float(from_time)),
            run_id=sql.Literal(run_id),
        )

    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching segment from {table_name}: {e}")
        return []


# ---------------------------------------------------------------------
# 3. Fetch N samples starting at a timestamp WITH optional run_id
# ---------------------------------------------------------------------
def fetch_sensor_batch(
    cursor, table_name, sample_count, start_time, run_id=None
):
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
