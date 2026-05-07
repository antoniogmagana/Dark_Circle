import asyncio
import os
import threading

import nats
import rclpy
from buffer import SensorBuffer
from dispatch import NATS_SUBJECT, load_channel_map, make_array_callback
from rclpy.node import Node
from std_msgs.msg import String


class IngestorNode(Node):
    """Subscribes to one ROS2 topic carrying bundled-channel JSON messages.

    The customer publishes a single ``std_msgs/String`` per sensor array per
    timestep; the JSON payload bundles all channels for that timestep.
    The channel-name → buffer-role mapping (and per-role expected rates)
    comes from a cluster-level YAML mounted into the pod, so the customer
    can change their channel-tag scheme without a code change.
    """

    def __init__(self, js, loop, channel_map, sensor_array, topic):
        # ROS2 node names allow alphanumerics and underscores but not hyphens,
        # while array IDs are RFC 1123 (hyphens, no underscores). Translate
        # for the node name only; sensor_array keeps its canonical form.
        node_name = "ingestor_" + sensor_array.replace("-", "_")
        super().__init__(node_name)
        self.js = js
        self.loop = loop
        self.sensor_array = sensor_array
        expected_rates = {spec.role: spec.expected_rate for spec in channel_map.values()}
        self.buffer = SensorBuffer(sensor_id=self.sensor_array, expected_rates=expected_rates)
        cb = make_array_callback(
            channel_map=channel_map,
            buffer=self.buffer,
            publish_payload=self._publish_payload,
        )
        self._sub = self.create_subscription(String, topic.strip(), cb, 10)

    def _publish_payload(self, payload):
        asyncio.run_coroutine_threadsafe(
            self.js.publish(NATS_SUBJECT, payload.SerializeToString()),
            self.loop,
        )


async def start_jetstream():
    nc = await nats.connect(os.environ["NATS_URL"])
    return nc, nc.jetstream()


def main():
    for var in ["NATS_URL", "SENSOR_ARRAY", "SENSOR_TOPIC"]:
        if var not in os.environ:
            raise OSError(f"Required environment variable '{var}' is not set")

    loop = asyncio.new_event_loop()
    nc, js = loop.run_until_complete(start_jetstream())
    sensor_array = os.environ["SENSOR_ARRAY"]
    topic = os.environ["SENSOR_TOPIC"]
    channel_map_path = os.environ.get("CHANNEL_MAP_PATH", "/etc/inference-engine/channels.yaml")
    channel_map = load_channel_map(channel_map_path)

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.init()
    node = IngestorNode(js, loop, channel_map, sensor_array, topic)
    rclpy.spin(node)
    rclpy.shutdown()
    asyncio.run_coroutine_threadsafe(nc.drain(), loop).result()


if __name__ == "__main__":
    main()
