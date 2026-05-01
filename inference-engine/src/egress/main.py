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

import nats
import rclpy
from inference_protos import inference_pb2
from rclpy.node import Node
from ros2_interfaces.msg import InferenceResult

QUEUE_GROUP = "egress"


class EgressNode(Node):
    def __init__(self):
        super().__init__("egressor")
        self.publisher = self.create_publisher(InferenceResult, "inference_result", 10)

    def _publish(self, payload):
        msg = InferenceResult()
        msg.sensor_id = payload.sensor_id
        msg.timestamp = payload.time_stamp.seconds + payload.time_stamp.nanos * 1e-9
        msg.vehicle_detected = payload.vehicle_detected
        msg.detection_confidence = payload.detection_confidence
        msg.vehicle_class = payload.vehicle_class
        msg.classification_confidence = payload.classification_confidence
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

    rclpy.init()
    node = EgressNode()
    # Two durable consumers, one per stream. The queue group keeps multiple
    # egress replicas (under KEDA) from publishing duplicate ROS2 messages.
    loop.run_until_complete(
        js.subscribe(
            "detection.result",
            queue=QUEUE_GROUP,
            durable="egress-detection",
            cb=node.on_detection,
            manual_ack=False,
        )
    )
    loop.run_until_complete(
        js.subscribe(
            "classification.result",
            queue=QUEUE_GROUP,
            durable="egress-classification",
            cb=node.on_classification,
            manual_ack=False,
        )
    )

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.spin(node)

    rclpy.shutdown()
    asyncio.run_coroutine_threadsafe(nc.drain(), loop).result()


if __name__ == "__main__":
    main()
