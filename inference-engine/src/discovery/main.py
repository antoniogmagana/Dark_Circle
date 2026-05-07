# Discovery Node
# Reads an expected-sensors ConfigMap each poll, checks the ROS2 graph for
# completeness, and spawns / tears down per-array Ingestor Deployments.
import os

import rclpy
from kubernetes import client
from kubernetes import config as k8s_config
from rclpy.node import Node
from whitelist import (
    ArraySpec,
    InvalidConfigError,
    PollState,
    build_ingestor_manifest,
    load_config,
)

SENSOR_MSG_TYPE = "std_msgs/msg/String"
POLL_INTERVAL = 5.0
GRACE_POLLS = 3
DEFAULT_CONFIG_PATH = "/etc/inference-engine/expected-sensors.yaml"


class DiscoveryNode(Node):
    def __init__(self, k8s_apps, namespace, template, config_path):
        super().__init__("discovery")
        self.k8s_apps = k8s_apps
        self.namespace = namespace
        self.template = template
        self.config_path = config_path
        self.state = PollState(grace_polls=GRACE_POLLS)
        self.active_arrays: set[str] = set()
        self._last_unknown: frozenset[str] = frozenset()

        self.timer = self.create_timer(POLL_INTERVAL, self._poll)

    def _sync_existing_deployments(self):
        """Reconcile in-memory active_arrays with what's actually in the cluster.

        Runs every poll, not just at startup, so Discovery notices when an
        Ingestor Deployment is deleted out-of-band (manual kubectl delete,
        namespace teardown, GitOps drift correction, etc.) and re-spawns it
        on the next poll. Without this, deleting a Deployment leaves
        active_arrays stale and the spawn pass skips the array forever.
        """
        deployments = self.k8s_apps.list_namespaced_deployment(
            namespace=self.namespace,
            label_selector="app=ingestor",
        )
        observed: set[str] = set()
        for d in deployments.items:
            sensor_array = d.metadata.labels.get("sensor-array")
            if sensor_array:
                observed.add(sensor_array)
                if sensor_array not in self.active_arrays:
                    self.get_logger().info(f"Found existing ingestor for {sensor_array}")

        vanished = self.active_arrays - observed
        for sensor_array in vanished:
            self.get_logger().warning(
                f"ingestor for {sensor_array} disappeared from cluster; will re-spawn"
            )

        self.active_arrays = observed

    def _load_config(self) -> dict[str, ArraySpec]:
        try:
            with open(self.config_path) as f:
                return load_config(f.read())
        except FileNotFoundError:
            self.get_logger().error(f"config file {self.config_path} not found; treating as empty")
            return {}
        except InvalidConfigError as exc:
            self.get_logger().error(f"invalid config, ignoring this poll: {exc}")
            return {}

    def _visible_topics(self) -> set[str]:
        return {
            topic for topic, types in self.get_topic_names_and_types() if SENSOR_MSG_TYPE in types
        }

    def _spawn(self, sensor_array: str, spec: ArraySpec):
        body = build_ingestor_manifest(self.template, sensor_array, spec)
        try:
            self.k8s_apps.create_namespaced_deployment(
                namespace=self.namespace,
                body=body,
            )
        except Exception as exc:
            self.get_logger().error(f"failed to spawn ingestor for {sensor_array}: {exc}")
            return
        self.active_arrays.add(sensor_array)
        self.get_logger().info(f"Spawned ingestor for {sensor_array}")

    def _teardown(self, sensor_array: str):
        try:
            self.k8s_apps.delete_namespaced_deployment(
                name=f"ingestor-{sensor_array}",
                namespace=self.namespace,
            )
        except Exception as exc:
            self.get_logger().error(f"failed to delete ingestor for {sensor_array}: {exc}")
            return
        self.active_arrays.discard(sensor_array)
        self.get_logger().info(f"Removed ingestor for {sensor_array}")

    def _poll(self):
        # Wrap the whole tick so one bad poll (transient K8s API error,
        # malformed config, ROS2 graph hiccup) doesn't kill the rclpy
        # timer. Without this, a single uncaught exception silently stops
        # the controller and Ingestors stop being reconciled.
        try:
            self._sync_existing_deployments()
            cfg = self._load_config()
            visible = self._visible_topics()
            decision = self.state.evaluate(
                config=cfg,
                visible=visible,
                active=self.active_arrays,
            )

            for array_id in decision.to_spawn:
                self._spawn(array_id, cfg[array_id])

            for array_id in decision.to_teardown:
                self._teardown(array_id)

            for array_id, missing in decision.log_awaiting.items():
                self.get_logger().info(f"awaiting {array_id}: missing {sorted(missing)}")

            self._log_unknown_topics(cfg, visible)
        except Exception as exc:
            self.get_logger().error(f"poll cycle failed: {exc!r}")

    def _log_unknown_topics(self, cfg: dict[str, ArraySpec], visible: set[str]):
        configured = {spec.topic for spec in cfg.values()}
        unknown = frozenset(visible - configured)
        if unknown != self._last_unknown:
            for topic in sorted(unknown - self._last_unknown):
                self.get_logger().info(f"ignoring unconfigured topic {topic}")
            self._last_unknown = unknown


def main():
    namespace = os.environ.get("NAMESPACE", "default")
    template_path = os.environ.get("TEMPLATE_PATH", "/app/ingestor-template.yaml")
    config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    with open(template_path) as f:
        template = f.read()

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    k8s_apps = client.AppsV1Api()

    rclpy.init()
    node = DiscoveryNode(k8s_apps, namespace, template, config_path)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
