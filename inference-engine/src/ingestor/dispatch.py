"""
Pure logic for SENSOR_ROLE_MAP parsing and per-role callback construction.

Free of rclpy / nats imports so it can be unit-tested without the runtime.
``ingestor.main`` imports from here and adds the ROS2 / NATS plumbing.
"""

import json
from collections.abc import Callable

NATS_SUBJECT = "sensor.data"

# Keys the SensorBuffer accepts as channel roles.
VALID_ROLES = frozenset({"acoustic", "seismic", "accel_x", "accel_y", "accel_z"})
REQUIRED_ROLES = frozenset({"acoustic", "seismic"})
ACCEL_ROLES = frozenset({"accel_x", "accel_y", "accel_z"})


class InvalidRoleMapError(ValueError):
    """SENSOR_ROLE_MAP is malformed or missing required roles."""


class UnknownRoleError(InvalidRoleMapError):
    """SENSOR_ROLE_MAP names a role the buffer does not understand."""


def parse_role_map(env_value: str) -> dict[str, str]:
    """Decode SENSOR_ROLE_MAP JSON and validate role coverage."""
    try:
        data = json.loads(env_value)
    except json.JSONDecodeError as exc:
        raise InvalidRoleMapError(f"SENSOR_ROLE_MAP is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise InvalidRoleMapError("SENSOR_ROLE_MAP must be a JSON object")

    roles = set(data.keys())

    unknown = roles - VALID_ROLES
    if unknown:
        raise UnknownRoleError(
            f"SENSOR_ROLE_MAP has unknown roles {sorted(unknown)}; "
            f"valid roles are {sorted(VALID_ROLES)}"
        )

    missing_required = REQUIRED_ROLES - roles
    if missing_required:
        raise InvalidRoleMapError(
            f"SENSOR_ROLE_MAP missing required roles {sorted(missing_required)}"
        )

    accel_present = roles & ACCEL_ROLES
    if accel_present and accel_present != ACCEL_ROLES:
        raise InvalidRoleMapError(
            "SENSOR_ROLE_MAP must include all of accel_x/y/z or none of them; "
            f"got {sorted(accel_present)}"
        )

    return data


def make_role_callback(
    role: str,
    buffer,
    publish_payload: Callable,
) -> Callable:
    """Build a ROS2 listener callback bound to ``role``.

    The returned callable forwards each incoming message into
    ``buffer.load_buffer(role, start_time, amplitude_readings)`` and, if the
    buffer returns a serializable payload, hands it off via
    ``publish_payload(payload)``.
    """
    counter = {"recv": 0, "publish": 0}

    def listener(msg):
        counter["recv"] += 1
        if counter["recv"] in (1, 10, 100, 1000) or counter["recv"] % 1000 == 0:
            print(
                f"[dispatch:{role}] received {counter['recv']} messages, "
                f"published {counter['publish']} windows",
                flush=True,
            )
        try:
            payload = buffer.load_buffer(role, msg.start_time, msg.amplitude_readings)
        except Exception as exc:
            print(f"[dispatch:{role}] buffer error: {exc!r}", flush=True)
            raise
        if payload is not None:
            counter["publish"] += 1
            print(
                f"[dispatch:{role}] PUBLISH window {counter['publish']} "
                f"(after {counter['recv']} messages)",
                flush=True,
            )
            publish_payload(payload)

    return listener
