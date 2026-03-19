import re
import psycopg2
from psycopg2 import sql


def sanitize_name(name, max_length=25):
    """Lowercase alphanumeric+underscore, no leading digits, capped length."""
    clean = re.sub(r"[^a-z0-9_]", "_", str(name).lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if clean and clean[0].isdigit():
        clean = f"v_{clean}"
    return (clean[:max_length]) or "unknown_entity"


def db_connect(db_conn_params):
    """Open a PostgreSQL connection with autocommit enabled."""
    try:
        conn = psycopg2.connect(**db_conn_params)
        conn.autocommit = True
        return conn, conn.cursor()
    except Exception as e:
        print(f"Failed to connect: {e}")
        raise


def db_close(conn, cursor):
    cursor.close()
    conn.close()


def get_time_bounds(cursor, table_name, run_id=None):
    """Return (min_timestamp, max_timestamp) for a table, optionally filtered by run_id."""
    if run_id is None:
        query = sql.SQL(
            "SELECT MIN(time_stamp), MAX(time_stamp) FROM {table}"
        ).format(table=sql.Identifier(table_name))
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


def fetch_sensor_batch(cursor, table_name, sample_count, start_time, run_id=None):
    """
    Fetch ``sample_count`` rows starting at ``start_time``.

    Handles audio/seismic (amplitude) and accelerometer (3-axis) tables.
    """
    target_cols = (
        sql.SQL("accel_x_ew, accel_y_ns, accel_z_ud")
        if "_accel_" in table_name
        else sql.SQL("amplitude")
    )

    if run_id is None:
        where = sql.SQL("time_stamp >= {t}").format(t=sql.Literal(start_time))
    else:
        where = sql.SQL("time_stamp >= {t} AND run_id = {r}").format(
            t=sql.Literal(start_time),
            r=sql.Literal(run_id),
        )

    query = sql.SQL(
        "SELECT {cols} FROM {table} WHERE {where} "
        "ORDER BY time_stamp ASC LIMIT {limit}"
    ).format(
        cols=target_cols,
        table=sql.Identifier(table_name),
        where=where,
        limit=sql.Literal(sample_count),
    )

    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []
