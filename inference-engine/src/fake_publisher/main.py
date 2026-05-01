"""
Fake ROS2 publisher for local smoke testing.

Emits ``RawSensorReading`` on configured topics so the Discovery node has
something to discover and the Ingestor has something to consume. Payload
sizes match the buffer's expected sample rates so the buffer's holding pen
doesn't drift unbounded:

  - acoustic (``aud``):           16000 samples/sec
  - seismic (``ehz``):                100 samples/sec
  - accel x/y/z (``ene/enn/enz``):    100 samples/sec each

Each topic publishes at FAKE_RATE_HZ ticks/sec, sending ``rate_hz_for_topic
/ FAKE_RATE_HZ`` samples per tick (so the integrated samples-per-second
stays correct).

Configuration (env):
  FAKE_TOPICS    Comma-separated list of topics to publish on.
  FAKE_RATE_HZ   Tick rate for the publisher loop. Default 10. Each topic
                 emits ``rate / FAKE_RATE_HZ`` samples per tick.
"""
import math
import os
import random
import time

import rclpy
from rclpy.node import Node

from ros2_interfaces.msg import RawSensorReading

# Sample rates per channel suffix, matching the buffer's defaults.
SAMPLE_RATES = {
    "aud": 16000,
    "ehz": 100,
    "ene": 100,
    "enn": 100,
    "enz": 100,
}


class FakePublisher(Node):
    def __init__(self, topics: list[str], rate_hz: float):
        super().__init__("fake_publisher")
        self.publishers_by_topic = {}
        period = 1.0 / rate_hz

        for topic in topics:
            suffix = topic.strip("/").split("/")[-1]
            sample_rate = SAMPLE_RATES.get(suffix)
            if sample_rate is None:
                self.get_logger().warning(
                    f"unknown topic suffix {suffix!r} for {topic}; "
                    f"defaulting to 100 samples/sec"
                )
                sample_rate = 100
            samples_per_tick = max(1, int(sample_rate / rate_hz))

            sensor_id = ".".join(topic.strip("/").split("/")[-2:])
            pub = self.create_publisher(RawSensorReading, topic, 10)
            self.publishers_by_topic[topic] = (pub, sensor_id, samples_per_tick)
            self.get_logger().info(
                f"publishing on {topic} (sensor_id={sensor_id}) "
                f"at {rate_hz} Hz x {samples_per_tick} samples = "
                f"{rate_hz * samples_per_tick:.0f} samples/sec"
            )
        self.create_timer(period, self._tick)

    def _tick(self):
        now = time.time()
        for topic, (pub, sensor_id, n) in self.publishers_by_topic.items():
            msg = RawSensorReading()
            msg.sensor_id = sensor_id
            msg.start_time = now
            msg.amplitude_readings = [
                int(1000 * math.sin(2 * math.pi * 5 * (i / max(n, 1)))
                    + random.randint(-50, 50))
                for i in range(n)
            ]
            pub.publish(msg)


def main():
    raw = os.environ.get("FAKE_TOPICS", "")
    topics = [t.strip() for t in raw.split(",") if t.strip()]
    if not topics:
        raise EnvironmentError("FAKE_TOPICS is required (comma-separated topics)")
    rate = float(os.environ.get("FAKE_RATE_HZ", "10"))

    rclpy.init()
    node = FakePublisher(topics, rate)
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
