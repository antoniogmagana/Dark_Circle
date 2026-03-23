# Discovery Node
# Watches the ROS2 network for active RawSensorReading topics
# Groups topics by sensor array prefix
# Spawns/destroys Ingestor deployments via the k8s API
import os
import yaml
import rclpy
from rclpy.node import Node
from collections import defaultdict
from kubernetes import client, config

SENSOR_MSG_TYPE = 'ros2_interfaces/msg/RawSensorReading'
POLL_INTERVAL   = 5.0   # seconds between ROS2 graph polls
GRACE_POLLS     = 3     # consecutive absent polls before teardown (~15 seconds)


class DiscoveryNode(Node):

    def __init__(self, k8s_apps, namespace, template):
        super().__init__('discovery')
        self.k8s_apps   = k8s_apps
        self.namespace  = namespace
        self.template   = template

        # tracks consecutive absent polls per sensor array
        self.absent_counts = defaultdict(int)

        # tracks which arrays currently have a running deployment
        self.active_arrays = set()

        # seed from deployments already running so we don't double-spawn on restart
        self._sync_existing_deployments()

        self.timer = self.create_timer(POLL_INTERVAL, self._poll)

    def _sync_existing_deployments(self):
        deployments = self.k8s_apps.list_namespaced_deployment(
            namespace=self.namespace,
            label_selector='app=ingestor'
        )
        for d in deployments.items:
            sensor_array = d.metadata.labels.get('sensor-array')
            if sensor_array:
                self.active_arrays.add(sensor_array)
                self.get_logger().info(f"Found existing ingestor for {sensor_array}")

    def _get_sensor_arrays(self):
        arrays = defaultdict(list)
        for topic, types in self.get_topic_names_and_types():
            if SENSOR_MSG_TYPE in types:
                parts = topic.strip('/').split('/')
                if len(parts) >= 2:
                    sensor_array = parts[0]
                    arrays[sensor_array].append(topic)
        return arrays

    def _spawn(self, sensor_array, topics):
        topics_str = ','.join(topics)
        manifest   = self.template.replace('<sensor_array_id>', sensor_array)
        manifest   = manifest.replace('<comma,separated,topics>', topics_str)
        body       = yaml.safe_load(manifest)
        self.k8s_apps.create_namespaced_deployment(
            namespace=self.namespace,
            body=body
        )
        self.active_arrays.add(sensor_array)
        self.get_logger().info(f"Spawned ingestor for {sensor_array}")

    def _teardown(self, sensor_array):
        self.k8s_apps.delete_namespaced_deployment(
            name=f'ingestor-{sensor_array}',
            namespace=self.namespace
        )
        self.active_arrays.discard(sensor_array)
        self.absent_counts.pop(sensor_array, None)
        self.get_logger().info(f"Removed ingestor for {sensor_array}")

    def _poll(self):
        visible_arrays = self._get_sensor_arrays()

        # spawn deployments for newly visible arrays
        for sensor_array, topics in visible_arrays.items():
            if sensor_array not in self.active_arrays:
                self._spawn(sensor_array, topics)
            else:
                self.absent_counts.pop(sensor_array, None)  # reset grace counter

        # increment absent counter for arrays no longer visible
        for sensor_array in list(self.active_arrays):
            if sensor_array not in visible_arrays:
                self.absent_counts[sensor_array] += 1
                self.get_logger().warn(
                    f"{sensor_array} absent for "
                    f"{self.absent_counts[sensor_array]}/{GRACE_POLLS} polls"
                )
                if self.absent_counts[sensor_array] >= GRACE_POLLS:
                    self._teardown(sensor_array)


def main():
    namespace     = os.environ.get("NAMESPACE", "default")
    template_path = os.environ.get("TEMPLATE_PATH", "/app/ingestor-template.yaml")

    with open(template_path) as f:
        template = f.read()

    try:
        config.load_incluster_config()       # running inside a k8s pod
    except config.ConfigException:
        config.load_kube_config()            # fallback for local dev

    k8s_apps = client.AppsV1Api()

    rclpy.init()
    node = DiscoveryNode(k8s_apps, namespace, template)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
