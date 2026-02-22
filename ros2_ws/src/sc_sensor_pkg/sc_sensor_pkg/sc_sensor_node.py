import rclpy
from rclpy.node import Node
import psycopg2
from psycopg2 import sql # Crucial for dynamic table names
from pipeline_interfaces.msg import RawSensorReading

class PostgresSensorNode(Node):
    def __init__(self):
        super().__init__('postgres_sensor_node')

        # 1. DB Connection Parameters
        self.declare_parameter('db_host', 'localhost')
        self.declare_parameter('db_port', 5432)
        self.declare_parameter('db_name', 'postgres')
        self.declare_parameter('db_user', 'postgres')
        self.declare_parameter('db_password', '')
        
        # 2. Query Filtering Parameters
        self.declare_parameter('target_table', 'sensor_data_1')
        self.declare_parameter('vehicle_id', 1)
        self.declare_parameter('sensor_id', 1)
        self.declare_parameter('publish_rate_hz', 10.0)

        # 3. Sensor Location Parameters
        self.declare_parameter('sensor_lat', 0.0)
        self.declare_parameter('sensor_lon', 0.0)

        # Retrieve parameters
        db_host = self.get_parameter('db_host').value
        db_port = self.get_parameter('db_port').value
        db_name = self.get_parameter('db_name').value
        db_user = self.get_parameter('db_user').value
        db_password = self.get_parameter('db_password').value
        
        table_name = self.get_parameter('target_table').value
        self.vehicle_id = self.get_parameter('vehicle_id').value
        self.sensor_id = self.get_parameter('sensor_id').value
        self.sensor_lat = self.get_parameter('sensor_lat').value
        self.sensor_lon = self.get_parameter('sensor_lon').value
        publish_rate = self.get_parameter('publish_rate_hz').value

        # Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(
                host=db_host, port=db_port, dbname=db_name, 
                user=db_user, password=db_password
            )
            self.cursor = self.conn.cursor()
            self.get_logger().info(f"Connected to DB. Querying vehicle {self.vehicle_id}, sensor {self.sensor_id} from {table_name}.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")
            raise

        # 4. Safely construct and execute the SQL query
        # We select time_stamp and amplitude. We use psycopg2.sql for the table identifier.
        query = sql.SQL(
            "SELECT time_stamp, amplitude FROM {table} "
            "WHERE vehicle_id = %s AND sensor_id = %s "
            "ORDER BY time_stamp ASC"
        ).format(table=sql.Identifier(table_name))

        self.cursor.execute(query, (self.vehicle_id, self.sensor_id))

        # Set up Publisher and Timer
        self.publisher_ = self.create_publisher(RawSensorReading, 'raw_sensor_stream', 10)
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        row = self.cursor.fetchone()
        
        if row is None:
            self.get_logger().info('End of database records.')
            self.timer.cancel()
            return

        db_time_stamp, amplitude = row

        # Create and populate the message
        msg = RawSensorReading()
        
        # Acting as a live sensor, we stamp it with the current ROS time
        msg.timestamp = self.get_clock().now().to_msg() 
        msg.sensor_label = f"v{self.vehicle_id}_s{self.sensor_id}"
        msg.reading = float(amplitude)
        msg.latitude = float(self.sensor_lat)
        msg.longitude = float(self.sensor_lon)

        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published: {msg.reading} at {msg.latitude}, {msg.longitude}')

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