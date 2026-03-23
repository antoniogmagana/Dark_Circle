import os
import rclpy
from rclpy.node import Node
import nats
import asyncio
import threading

# locat packages
from buffer import SensorBuffer
from ros2_interfaces.msg import RawSensorReading



CHANNEL_MAP = {
    'aud': 'acoustic',
    'ehz': 'seismic',
    'ene': 'accel_x',
    'enn': 'accel_y',
    'enz': 'accel_z'
}

NATS_SUBJECT = "sensor.data"

class IngestorNode(Node):
    """ 
    Ingestor ROS2 Node Class
    Takes in NATS connection and loop structure, 
    listens for ROS2 topics,
    publishes ingested data to NATS queue
    """
    def __init__(self, nc, loop, topics, sensor_array):
        """
        constructor
        """
        super().__init__(f'ingestor_{sensor_array}')
        self.nc = nc
        self.loop = loop
        self.sensor_array = sensor_array
        self.buffer = SensorBuffer(sensor_id=self.sensor_array)
        self.subscriptions = []
        for topic in topics:
            self.subscriptions.append(
                self.create_subscription(
                    RawSensorReading,
                    topic.strip(),
                    self.listener_callback,
                    10
                )
            )

    def listener_callback(self, msg):
        """
        combined listener and publisher
        """
        channel_code = msg.sensor_id.split('.')[-1]
        channel = CHANNEL_MAP.get(channel_code)
        if channel is None:
            return
        
        payload = self.buffer.load_buffer(
            channel, msg.amplitude_readings, msg.start_time
        )

        if payload is not None:
            asyncio.run_coroutine_threadsafe(
                self.nc.publish(NATS_SUBJECT, payload.SerializeToString()),
                self.loop
            )


async def start_nats():
    return await nats.connect(os.environ["NATS_URL"])


def main():
    for var in ["NATS_URL", "SENSOR_ARRAY", "SENSOR_TOPICS"]:
        if var not in os.environ:
            raise EnvironmentError(f"Required environment variable '{var}' is not set")
    
    loop = asyncio.new_event_loop()
    nc = loop.run_until_complete(start_nats())
    topics = os.environ["SENSOR_TOPICS"].split(',')
    sensor_array = os.environ["SENSOR_ARRAY"]

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.init()
    node = IngestorNode(nc, loop, topics, sensor_array)
    rclpy.spin(node)
    rclpy.shutdown()
    loop.run_until_complete(nc.drain())


if __name__ == '__main__':
    main()