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

def fetch_sensor_batch(cursor, base_name, vehicle, sensor, offset=0):
    """
    Fetches exactly one DB_CHUNK_SIZE of data.
    Automatically handles single-axis (amplitude) or tri-axial (x, y, z) schemas.
    """
    table_name = f"{base_name}_{sanitize_name(vehicle)}_{sanitize_name(sensor)}"
    
    # Route the columns based on your load_db.py schema definitions
    if base_name == "accel":
        target_cols = "accel_x_ew, accel_y_ns, accel_z_ud"
    else:
        target_cols = "amplitude"

    query = sql.SQL(
        "SELECT {columns} FROM {table} ORDER BY time_stamp ASC LIMIT {limit} OFFSET {offset}"
    ).format(
        columns=sql.SQL(target_cols), # sql.SQL allows raw comma-separated column strings
        table=sql.Identifier(table_name),
        limit=sql.Literal(DB_CHUNK_SIZE),
        offset=sql.Literal(offset)
    )
    
    try:
        cursor.execute(query)
        # Returns a list of tuples. 
        # For audio: [(0.12,), (0.15,)]
        # For accel: [(0.1, 0.2, 0.9), (0.1, 0.3, 0.8)]
        return cursor.fetchall() 
        
    except (Exception, psycopg2.Error) as e:
        print(f"Error fetching data from {table_name}: {e}")
        return []