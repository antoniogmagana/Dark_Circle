import psycopg2
import variables


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


def get_vehicle_id(conn, cursor, vehicle_name):
    # 1. Check if it exists
    check_query = "SELECT vehicle_id FROM vehicle_ids WHERE vehicle = %s"
    try:
        cursor.execute(check_query, (vehicle_name,))
        result = cursor.fetchone()
        return result[0]
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking vehicle_id for {vehicle_name}: {e}")
        return None


def get_sensor_id(conn, cursor, sensor_name):
    # 1. Check if it exists
    check_query = "SELECT sensor_id FROM sensor_ids WHERE sensor = %s"
    try:
        cursor.execute(check_query, (sensor_name,))
        result = cursor.fetchone()
        return result[0]
    except (Exception, psycopg2.Error) as e:
        print(f"Error checking sensor_id for {sensor_name}: {e}")
        return None
