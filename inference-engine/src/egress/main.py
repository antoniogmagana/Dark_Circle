"""Egress node: NATS -> ROS2 bridge.

One ROS2 message per inference window:
- ``classification.result`` carries the full result for positives
  (vehicle_class + both confidences). Published as-is.
- ``detection.result`` is consumed only to surface negatives — windows
  where no vehicle was detected. Positives are skipped here because the
  classifier emits the authoritative payload on ``classification.result``;
  publishing both would duplicate every positive on /inference_result.
"""

import asyncio
import os
import threading
import time
from pathlib import Path

import nats
import rclpy
from inference_protos import inference_pb2
from rclpy.node import Node
from ros2_interfaces.msg import InferenceResult

DEFAULT_OUTPUT_TOPIC = "/inference_result"
READY_SENTINEL = Path(os.environ.get("READY_SENTINEL", "/tmp/ready"))


class EgressNode(Node):
    def __init__(self, output_topic: str):
        super().__init__("egressor")
        self.publisher = self.create_publisher(InferenceResult, output_topic, 10)

    def _publish(self, payload):
        capture_time = payload.time_stamp.seconds + payload.time_stamp.nanos * 1e-9
        publish_time = time.time()
        latency = publish_time - capture_time

        payload.publish_time = publish_time
        payload.latency_seconds = latency

        msg = InferenceResult()
        msg.sensor_id = payload.sensor_id
        msg.timestamp = capture_time
        msg.vehicle_detected = payload.vehicle_detected
        msg.detection_confidence = payload.detection_confidence
        msg.vehicle_class = payload.vehicle_class
        msg.classification_confidence = payload.classification_confidence
        msg.publish_time = publish_time
        msg.latency_seconds = latency
        self.publisher.publish(msg)

    async def on_detection(self, msg):
        result = inference_pb2.DetectionResult()
        result.ParseFromString(msg.data)
        if result.vehicle_detected:
            return
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = result.sensor_data.sensor_id
        payload.time_stamp.CopyFrom(result.sensor_data.time_stamp)
        payload.vehicle_detected = False
        payload.detection_confidence = result.confidence
        self._publish(payload)

    async def on_classification(self, msg):
        payload = inference_pb2.EgressPayload()
        payload.ParseFromString(msg.data)
        self._publish(payload)


async def start_nats():
    return await nats.connect(os.environ["NATS_URL"])


def main():
    if "NATS_URL" not in os.environ:
        raise OSError("Required environment variable 'NATS_URL' is not set")

    loop = asyncio.new_event_loop()
    nc = loop.run_until_complete(start_nats())
    js = nc.jetstream()

    output_topic = os.environ.get("INFERENCE_RESULT_TOPIC", DEFAULT_OUTPUT_TOPIC)

    rclpy.init()
    node = EgressNode(output_topic)

    # Two durable push consumers, pre-created by jetstream-init with
    # deliver_group="egress" so additional egress replicas share work via
    # the queue group instead of each receiving every message. Binding
    # rather than auto-creating sidesteps nats-py's client-side check
    # that rejects ``queue=X, durable=Y`` when X != Y, which we can't
    # satisfy here because we need one queue group across two streams.
    async def bind(stream: str, consumer: str, cb):
        info = await js.consumer_info(stream, consumer)
        return await js.subscribe_bind(
            stream=stream,
            consumer=consumer,
            config=info.config,
            cb=cb,
            manual_ack=False,
        )

    loop.run_until_complete(bind("DETECTION_RESULT", "egress-detection", node.on_detection))
    loop.run_until_complete(bind("CLASSIFICATION_RESULT", "egress-classification", node.on_classification))

    READY_SENTINEL.touch()
    print(
        f"[egress] READY: NATS subscribed, ROS2 publishing. "
        f"Inference results will appear on {output_topic}.",
        flush=True,
    )

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.spin(node)

    rclpy.shutdown()
    asyncio.run_coroutine_threadsafe(nc.drain(), loop).result()


if __name__ == "__main__":
    main()
