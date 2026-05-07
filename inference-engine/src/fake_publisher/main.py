"""
Fake ROS2 publisher for local smoke testing and customer pipeline validation.

Emits ``std_msgs/String`` on configured per-array topics, with a JSON
payload bundling all channels for one timestep — the same wire format the
customer's real publisher uses. This lets the customer drive the full
ingestor → detect → classify → egress pipeline without their hardware.

Each tick the publisher emits, per topic, one JSON document:

    {"sensor_id": "<id>", "state": "background"|"trigger",
     "timestamp_unix": <float>, "timestamp_utc": "<iso>",
     "channels": [{"channel": "<tag>", "sampling_rate": <int>,
                   "dt": <float>, "readings": [...]}, ...]}

Per-channel ``readings`` length is ``sampling_rate / FAKE_RATE_HZ`` so the
integrated samples-per-second matches the configured rate, keeping the
ingestor's buffer aligned without holding-pen drift.

Configuration (env):
  FAKE_TOPICS         Comma-separated list of per-array topics. e.g.
                      ``/shake_001/data,/shake_002/data``.
  FAKE_RATE_HZ        Tick rate for the publisher loop. Default 10.
  FAKE_TRIGGER_EVERY  Emit ``state="trigger"`` every Nth tick; rest are
                      ``"background"``. Default 10. Set to 0 to always
                      emit ``"background"``.
  FAKE_CHANNEL_TAGS   Comma-separated channel tags (default
                      ``MIC,EHZ,ENE,ENN,ENZ``). Match the cluster's
                      channels.yaml so the ingestor recognizes them.
"""

import datetime as _dt
import json
import math
import os
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Default per-tag sample rates. Customer-tag-keyed; override via
# FAKE_CHANNEL_TAGS + matching cluster channels.yaml if your fleet differs.
DEFAULT_SAMPLE_RATES = {
    "MIC": 16000,
    "EHZ": 100,
    "ENE": 100,
    "ENN": 100,
    "ENZ": 100,
}


def _channel_tags() -> list[str]:
    raw = os.environ.get("FAKE_CHANNEL_TAGS", "")
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags or list(DEFAULT_SAMPLE_RATES.keys())


def _sample_rate_for(tag: str) -> int:
    return DEFAULT_SAMPLE_RATES.get(tag, 100)


class FakePublisher(Node):
    def __init__(
        self,
        topics: list[str],
        rate_hz: float,
        channel_tags: list[str],
        trigger_every: int,
    ):
        super().__init__("fake_publisher")
        self.publishers_by_topic = {}
        self.channel_tags = channel_tags
        self.trigger_every = trigger_every
        self._tick_count = 0
        period = 1.0 / rate_hz

        # samples_per_tick per channel, fixed for all topics.
        self.samples_per_tick = {
            tag: max(1, int(_sample_rate_for(tag) / rate_hz)) for tag in channel_tags
        }

        for topic in topics:
            sensor_id = topic.strip("/").replace("/", ".")
            pub = self.create_publisher(String, topic, 10)
            self.publishers_by_topic[topic] = (pub, sensor_id)
            self.get_logger().info(
                f"publishing on {topic} (sensor_id={sensor_id}) "
                f"at {rate_hz} Hz; channels="
                + ", ".join(
                    f"{tag}@{_sample_rate_for(tag)}Hz x{self.samples_per_tick[tag]}/tick"
                    for tag in channel_tags
                )
            )
        self.create_timer(period, self._tick)

    def _build_readings(self, tag: str, n: int) -> list[float]:
        # Synthetic: 5 Hz sine + uniform noise. float32-typed values.
        return [
            float(1000.0 * math.sin(2 * math.pi * 5 * (i / max(n, 1))) + random.uniform(-50.0, 50.0))
            for i in range(n)
        ]

    def _tick(self):
        self._tick_count += 1
        now = time.time()
        utc = _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc).isoformat()
        is_trigger = self.trigger_every > 0 and (self._tick_count % self.trigger_every == 0)
        state = "trigger" if is_trigger else "background"

        for _topic, (pub, sensor_id) in self.publishers_by_topic.items():
            doc = {
                "sensor_id": sensor_id,
                "state": state,
                "timestamp_unix": now,
                "timestamp_utc": utc,
                "channels": [
                    {
                        "channel": tag,
                        "sampling_rate": _sample_rate_for(tag),
                        "dt": 1.0 / _sample_rate_for(tag),
                        "readings": self._build_readings(tag, self.samples_per_tick[tag]),
                    }
                    for tag in self.channel_tags
                ],
            }
            msg = String()
            msg.data = json.dumps(doc)
            pub.publish(msg)


def main():
    raw = os.environ.get("FAKE_TOPICS", "")
    topics = [t.strip() for t in raw.split(",") if t.strip()]
    if not topics:
        raise OSError("FAKE_TOPICS is required (comma-separated topics)")
    rate = float(os.environ.get("FAKE_RATE_HZ", "10"))
    trigger_every = int(os.environ.get("FAKE_TRIGGER_EVERY", "10"))
    channel_tags = _channel_tags()

    rclpy.init()
    node = FakePublisher(topics, rate, channel_tags, trigger_every)
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
