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


def _build_where(start_time=None, run_id=None, scene_id=None):
    """Build a WHERE clause from optional filters."""
    conditions = []

    if start_time is not None:
        conditions.append(
            sql.SQL("time_stamp >= {t}").format(t=sql.Literal(start_time))
        )
    if run_id is not None:
        conditions.append(
            sql.SQL("run_id = {r}").format(r=sql.Literal(run_id))
        )
    if scene_id is not None:
        conditions.append(
            sql.SQL("scene_id = {s}").format(s=sql.Literal(scene_id))
        )

    if not conditions:
        return sql.SQL("")

    return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)


def get_time_bounds(cursor, table_name, run_id=None, scene_id=None):
    """
    Return (min_timestamp, max_timestamp) for a table,
    optionally filtered by run_id and/or scene_id.
    """
    where = _build_where(run_id=run_id, scene_id=scene_id)

    query = sql.SQL(
        "SELECT MIN(time_stamp), MAX(time_stamp) FROM {table}{where}"
    ).format(
        table=sql.Identifier(table_name),
        where=where,
    )

    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] or 0.0, result[1] or 0.0
    except Exception as e:
        print(f"Error fetching bounds for {table_name}: {e}")
        return 0.0, 0.0


def fetch_sensor_batch(cursor, table_name, sample_count, start_time,
                       run_id=None, scene_id=None):
    """
    Fetch ``sample_count`` rows starting at ``start_time``.

    Handles audio/seismic (amplitude) and accelerometer (3-axis) tables.
    Optionally filters by run_id and/or scene_id.
    """
    target_cols = (
        sql.SQL("accel_x_ew, accel_y_ns, accel_z_ud")
        if "_accel_" in table_name
        else sql.SQL("amplitude")
    )

    where = _build_where(start_time=start_time, run_id=run_id, scene_id=scene_id)

    query = sql.SQL(
        "SELECT {cols} FROM {table}{where} "
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