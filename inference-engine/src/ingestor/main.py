import asyncio
import os
import threading

import nats
import rclpy
from buffer import SensorBuffer
from dispatch import NATS_SUBJECT, make_role_callback, parse_role_map
from rclpy.node import Node
from ros2_interfaces.msg import RawSensorReading


class IngestorNode(Node):
    """Subscribes to one ROS2 topic per configured role.

    Each subscription's callback is bound at construction time to the role
    it serves (``acoustic`` / ``seismic`` / ``accel_x|y|z``). Discovery
    passed the binding in via ``SENSOR_ROLE_MAP``, so we never re-derive
    the role from message content.
    """

    def __init__(self, js, loop, role_map, sensor_array):
        # ROS2 node names allow alphanumerics and underscores but not hyphens,
        # while array IDs are RFC 1123 (hyphens, no underscores). Translate
        # for the node name only; sensor_array keeps its canonical form.
        node_name = "ingestor_" + sensor_array.replace("-", "_")
        super().__init__(node_name)
        self.js = js
        self.loop = loop
        self.sensor_array = sensor_array
        self.buffer = SensorBuffer(sensor_id=self.sensor_array)
        # ``subscriptions`` is reserved on rclpy Node; use a different name
        # to keep strong references to per-role subscriptions alive.
        self._role_subs = []

        for role, topic in role_map.items():
            cb = make_role_callback(
                role=role,
                buffer=self.buffer,
                publish_payload=self._publish_payload,
            )
            self._role_subs.append(
                self.create_subscription(RawSensorReading, topic.strip(), cb, 10)
            )

    def _publish_payload(self, payload):
        asyncio.run_coroutine_threadsafe(
            self.js.publish(NATS_SUBJECT, payload.SerializeToString()),
            self.loop,
        )


async def start_jetstream():
    nc = await nats.connect(os.environ["NATS_URL"])
    return nc, nc.jetstream()


def main():
    for var in ["NATS_URL", "SENSOR_ARRAY", "SENSOR_ROLE_MAP"]:
        if var not in os.environ:
            raise OSError(f"Required environment variable '{var}' is not set")

    loop = asyncio.new_event_loop()
    nc, js = loop.run_until_complete(start_jetstream())
    role_map = parse_role_map(os.environ["SENSOR_ROLE_MAP"])
    sensor_array = os.environ["SENSOR_ARRAY"]

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.init()
    node = IngestorNode(js, loop, role_map, sensor_array)
    rclpy.spin(node)
    rclpy.shutdown()
    asyncio.run_coroutine_threadsafe(nc.drain(), loop).result()


if __name__ == "__main__":
    main()
