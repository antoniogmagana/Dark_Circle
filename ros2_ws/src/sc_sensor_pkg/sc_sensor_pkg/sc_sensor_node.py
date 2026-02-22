import rclpy
from rclpy.node import Node
import psycopg2
from psycopg2 import sql
from pipeline_interfaces.msg import RawSensorReading

class PostgresSensorNode(Node):
    def __init__(self):
        super().__init__('postgres_sensor_node')

        # 1. Parameters
        # DB Connection
        self.declare_parameter('db_host', 'localhost')
        self.declare_parameter('db_port', 5432)
        self.declare_parameter('db_name', 'postgres')
        self.declare_parameter('db_user', 'postgres')
        self.declare_parameter('db_password', '')
        
        # Table variables
        self.declare_parameter('target_table', 'vehicle_type_rsx')
        self.declare_parameter('sensor_lat', 0.0)
        self.declare_parameter('sensor_lon', 0.0)

        # Retrieve parameters
        db_host = self.get_parameter('db_host').value
        db_port = self.get_parameter('db_port').value
        db_name = self.get_parameter('db_name').value
        db_user = self.get_parameter('db_user').value
        db_password = self.get_parameter('db_password').value
        
        table_name = self.get_parameter('target_table').value
        self.sensor_id = table_name.split("_")[-1]
        self.sensor_lat = self.get_parameter('sensor_lat').value
        self.sensor_lon = self.get_parameter('sensor_lon').value

        # 2. Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(
                host=db_host, port=db_port, dbname=db_name, 
                user=db_user, password=db_password
            )
            self.cursor = self.conn.cursor()
            self.get_logger().info(f"Connected to DB. Querying {table_name}...")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            raise

        # 3. Safely execute the SQL query
        query = sql.SQL(
            "SELECT time_stamp, amplitude FROM {table} "
            "ORDER BY time_stamp ASC"
        ).format(table=sql.Identifier(table_name))

        self.cursor.execute(query, (self.vehicle_id, self.sensor_id))

        # 4. Initialize Publisher and state tracking
        self.publisher_ = self.create_publisher(RawSensorReading, 'raw_sensor_stream', 10)
        
        # Pre-fetch the very first row to kick off the cycle
        self.next_row = self.cursor.fetchone()
        
        if self.next_row:
            # Trigger the first callback immediately (0.0001 seconds)
            self.timer = self.create_timer(0.0001, self.dynamic_timer_callback)
        else:
            self.get_logger().info("No data found for the specified parameters.")

    def dynamic_timer_callback(self):
        # 1. Cancel the timer that just woke us up
        self.timer.cancel()
        
        if self.next_row is None:
            self.get_logger().info('End of database records.')
            return

        # 2. Extract current row data
        current_db_timestamp, amplitude = self.next_row

        # 3. Create and publish the message
        msg = RawSensorReading()
        msg.timestamp = self.get_clock().now().to_msg() 
        msg.sensor_label = str(self.sensor_id)
        msg.reading = float(amplitude)
        msg.latitude = float(self.sensor_lat)
        msg.longitude = float(self.sensor_lon)

        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published {msg.sensor_label} : {msg.reading}')

        # 4. Fetch the NEXT row to calculate the delay
        self.next_row = self.cursor.fetchone()
        
        if self.next_row:
            next_db_timestamp = self.next_row[0]
            
            # Calculate the time gap in seconds
            wait_time_seconds = float(next_db_timestamp - current_db_timestamp)
            
            # Failsafe: If timestamps are identical or negative due to bad data, 
            # force a tiny delay so the node doesn't freeze or throw a timer error.
            wait_time_seconds = max(0.0001, wait_time_seconds)
            
            # 5. Schedule the next publish exactly 'wait_time_seconds' from now
            self.timer = self.create_timer(wait_time_seconds, self.dynamic_timer_callback)
        else:
            self.get_logger().info('Final record published. Shutting down stream.')


def main(args=None):
    rclpy.init(args=args)
    node = PostgresSensorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cursor.close()
        node.conn.close()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()