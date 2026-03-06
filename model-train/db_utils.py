import psycopg2
from psycopg2 import sql
import re
from config import DB_CONN_PARAMS, DB_CHUNK_SIZE

def sanitize_name(name, max_length=25):
    """Replicated exactly from load_db.py to ensure perfect table name matching."""
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

def fetch_sensor_batch(cursor, base_name, vehicle, sensor, offset=0, limit=16000):
    """
    Fetches a precise window of data.
    Automatically handles single-axis (amplitude) or tri-axial (x, y, z) schemas.
    """
    table_name = f"{base_name}_{sanitize_name(vehicle)}_{sanitize_name(sensor)}"
    
    if base_name == "accel":
        target_cols = "accel_x_ew, accel_y_ns, accel_z_ud"
    else:
        target_cols = "amplitude"

    query = sql.SQL(
        "SELECT {columns} FROM {table} ORDER BY time_stamp ASC LIMIT {limit} OFFSET {offset}"
    ).format(
        columns=sql.SQL(target_cols),
        table=sql.Identifier(table_name),
        limit=sql.Literal(limit), # <--- CHANGED THIS LINE
        offset=sql.Literal(offset)
    )
    
    try:
        cursor.execute(query)
        return cursor.fetchall() 
        
    except (Exception, psycopg2.Error) as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []