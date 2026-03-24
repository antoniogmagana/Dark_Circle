import os
import asyncio
import rclpy
from rclpy.node import Node
import nats
import threading

# local imports
from inference_protos import inference_pb2
from ros2_interfaces.msg import InferenceResult


class EgressNode(Node):
    def __init__(self, nc, loop, sensor_array):
        super().__init__(f'egressor_{sensor_array}')
        self.nc = nc
        self.loop = loop
        self.sensor_array = sensor_array
        self.publisher = self.create_publisher(InferenceResult, 'inference_result', 10)

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
        payload = inference_pb2.EgressPayload()
        payload.sensor_id = result.sensor_data.sensor_id
        payload.time_stamp.CopyFrom(result.sensor_data.time_stamp)
        payload.vehicle_detected = result.vehicle_detected
        payload.detection_confidence = result.confidence
        if not result.vehicle_detected:
            return
        self._publish(payload)

    async def on_classification(self, msg):
        payload = inference_pb2.EgressPayload()
        payload.ParseFromString(msg.data)
        self._publish(payload)

async def start_nats():
    return await nats.connect(os.environ["NATS_URL"])

def main():
    for var in ["NATS_URL", "SENSOR_ARRAY"]:
        if var not in os.environ:
            raise EnvironmentError(f"Required environment variable '{var}' is not set")
    
    loop = asyncio.new_event_loop()
    nc = loop.run_until_complete(start_nats())
    sensor_array = os.environ["SENSOR_ARRAY"]
    node = EgressNode(nc, loop, sensor_array)
    loop.run_until_complete(nc.subscribe("detection.result", cb=node.on_detection))
    loop.run_until_complete(nc.subscribe("classification.result", cb=node.on_classification))

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    rclpy.init()
    rclpy.spin(node)

    rclpy.shutdown()
    loop.run_until_complete(nc.drain())


if __name__ == '__main__':
    main()