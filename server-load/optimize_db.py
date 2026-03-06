import psycopg2
import variables  # Pulling DB_CONN_PARAMS/conn_params from your local file


def apply_global_timestamp_indices():
    """
    Finds every table in the database that has a 'time_stamp' column
    and creates a B-Tree index for it to accelerate training.
    """
    try:
        # Using variables.conn_params as defined in your load_db.py setup
        conn = psycopg2.connect(**variables.conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        print("Connected to PostgreSQL for optimization.")
    except Exception as e:
        print(f"Failed to connect using variables.py parameters: {e}")
        return

    # Query the information_schema to find all tables with a 'time_stamp' column.
    # This covers iobt, focal, and m3nvc regardless of their naming convention.
    cursor.execute(
        """
        SELECT table_name 
        FROM information_schema.columns 
        WHERE column_name = 'time_stamp' 
        AND table_schema = 'public';
    """
    )

    target_tables = [row[0] for row in cursor.fetchall()]

    if not target_tables:
        print("No tables found with a 'time_stamp' column.")
        cursor.close()
        conn.close()
        return

    print(f"Found {len(target_tables)} tables requiring optimization. Starting...")

    for i, table in enumerate(target_tables):
        try:
            # 1. Create the B-Tree index
            # This allows O(log n) lookups instead of full table scans during training
            index_name = f"idx_{table}_timestamp"
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} (time_stamp);"
            )

            # 2. Analyze the table
            # This updates the query planner statistics so it actually uses the new index
            cursor.execute(f"ANALYZE {table};")

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(target_tables)} tables indexed.")

        except Exception as e:
            print(f"Could not index {table}: {e}")
            conn.rollback()

    print(f"\nOptimization Complete. {len(target_tables)} tables are now indexed.")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    apply_global_timestamp_indices()
