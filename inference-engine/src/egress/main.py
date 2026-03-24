import os
import asyncio
import rclpy
from rclpy.node import Node
import nats

# local imports
from inference_protos import inference_pb2
from ros2_intergaces.msg import InferenceResult


class EgressNode(Node):
    def __init__(self, nc, loop, sensor_array):
        super().__init__(f'egressor_{sensor_array}')
        self.nc = nc
        self.loop = loop
        self.sensor_array = sensor_array
        self.publisher = self.create_publisher(InferenceResult, 'inference_result', 10)

    def _publish(self, payload):
        pass

async def on_detection(msg):
    pass

async def on_classificaiton(msg):
    pass

def main():
    pass

if __name__ == '__main__':
    main()