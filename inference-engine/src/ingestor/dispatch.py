"""
Channel-map loading and per-message ROS2 callback construction.

The Ingestor subscribes to one ``std_msgs/String`` topic per sensor array.
Each message is a JSON document bundling all channels for one timestep.
This module owns:

  * ``load_channel_map`` — parses the cluster-level YAML file that maps the
    customer's free-form channel-tag strings (e.g. ``"MIC"``, ``"EHZ"``)
    onto the SensorBuffer's role names (``"acoustic"``, ``"seismic"``,
    ``"accel_x|y|z"``) and pins each role's expected sampling rate.
  * ``make_array_callback`` — builds the rclpy listener that parses the JSON,
    fans channel slices out into ``buffer.load_buffer``, and forwards any
    completed window to NATS.

Free of rclpy / nats imports so it can be unit-tested without the runtime.
``ingestor.main`` imports from here and adds the ROS2 / NATS plumbing.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import yaml
from buffer import ACCEL_ROLES, REQUIRED_ROLES, VALID_ROLES

NATS_SUBJECT = "sensor.data"


class InvalidChannelMapError(ValueError):
    """channels.yaml is malformed or fails role-coverage validation."""


@dataclass(frozen=True)
class ChannelSpec:
    """One row of the channel map: customer tag → buffer role + expected rate."""

    role: str
    expected_rate: int


def load_channel_map(path: str) -> dict[str, ChannelSpec]:
    """Parse ``channels.yaml`` into ``{customer_tag: ChannelSpec}``.

    Validates that the resulting role set covers ``acoustic`` and ``seismic``
    (the buffer's REQUIRED_ROLES) and that any accel role implies all three.
    """
    try:
        with open(path) as f:
            raw = yaml.safe_load(f.read())
    except FileNotFoundError as exc:
        raise InvalidChannelMapError(f"channel map file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise InvalidChannelMapError(f"malformed YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or "channels" not in raw:
        raise InvalidChannelMapError(
            f"{path}: must define a top-level 'channels' mapping"
        )

    entries = raw["channels"] or {}
    if not isinstance(entries, dict):
        raise InvalidChannelMapError(f"{path}: 'channels' must be a mapping")

    spec_by_tag: dict[str, ChannelSpec] = {}
    for tag, body in entries.items():
        if not isinstance(tag, str) or not tag:
            raise InvalidChannelMapError(f"{path}: channel tags must be non-empty strings")
        if not isinstance(body, dict):
            raise InvalidChannelMapError(f"{path}: entry for {tag!r} must be a mapping")
        role = body.get("role")
        rate = body.get("expected_rate")
        if role not in VALID_ROLES:
            raise InvalidChannelMapError(
                f"{path}: tag {tag!r} maps to unknown role {role!r}; "
                f"valid roles are {sorted(VALID_ROLES)}"
            )
        if not isinstance(rate, int) or rate <= 0:
            raise InvalidChannelMapError(
                f"{path}: tag {tag!r} expected_rate must be a positive integer, got {rate!r}"
            )
        spec_by_tag[tag] = ChannelSpec(role=role, expected_rate=rate)

    roles_present = {spec.role for spec in spec_by_tag.values()}
    missing_required = REQUIRED_ROLES - roles_present
    if missing_required:
        raise InvalidChannelMapError(
            f"{path}: channel map missing required roles {sorted(missing_required)}"
        )
    accel_present = roles_present & ACCEL_ROLES
    if accel_present and accel_present != ACCEL_ROLES:
        raise InvalidChannelMapError(
            f"{path}: accel must be all of {sorted(ACCEL_ROLES)} or none; "
            f"got {sorted(accel_present)}"
        )

    return spec_by_tag


def make_array_callback(
    channel_map: dict[str, ChannelSpec],
    buffer,
    publish_payload: Callable,
) -> Callable:
    """Build the rclpy listener for one sensor-array topic.

    The returned callable parses each ``std_msgs/String`` message as a JSON
    document, calls ``buffer.maybe_close_window(timestamp_unix)`` once, and
    fans the channel readings into ``buffer.load_buffer(role, timestamp,
    readings)`` per entry. Completed windows are forwarded to NATS via
    ``publish_payload``.

    Soft-validates per-message ``sampling_rate`` against the configured
    ``expected_rate`` for that role: a mismatch logs a warning but processing
    continues. Hard rate validation (drop the window) lives in
    ``SensorBuffer._package_window`` at window close.

    Unknown channel tags are skipped with a log line; malformed JSON or
    missing required fields drops the whole message.
    """
    counter = {"recv": 0, "publish": 0, "background": 0, "trigger": 0, "unknown_state": 0}

    def listener(msg):
        counter["recv"] += 1
        if counter["recv"] in (1, 10, 100, 1000) or counter["recv"] % 1000 == 0:
            print(
                f"[dispatch:array] received {counter['recv']} messages, "
                f"published {counter['publish']} windows "
                f"(state: bg={counter['background']} trig={counter['trigger']} "
                f"unk={counter['unknown_state']})",
                flush=True,
            )

        try:
            doc = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"[dispatch:array] dropped malformed JSON: {exc!r}", flush=True)
            return

        if not isinstance(doc, dict):
            print("[dispatch:array] dropped: top-level JSON not an object", flush=True)
            return

        state = doc.get("state")
        if state == "background":
            counter["background"] += 1
        elif state == "trigger":
            counter["trigger"] += 1
        else:
            counter["unknown_state"] += 1

        ts = doc.get("timestamp_unix")
        if not isinstance(ts, (int, float)):
            print(
                f"[dispatch:array] dropped: timestamp_unix missing or non-numeric ({ts!r})",
                flush=True,
            )
            return
        ts = float(ts)

        channels = doc.get("channels")
        if not isinstance(channels, list):
            print("[dispatch:array] dropped: 'channels' missing or not a list", flush=True)
            return

        try:
            payload = buffer.maybe_close_window(ts)
        except Exception as exc:
            print(f"[dispatch:array] buffer error in maybe_close_window: {exc!r}", flush=True)
            raise
        if payload is not None:
            counter["publish"] += 1
            print(
                f"[dispatch:array] PUBLISH window {counter['publish']} "
                f"(after {counter['recv']} messages)",
                flush=True,
            )
            publish_payload(payload)

        for entry in channels:
            if not isinstance(entry, dict):
                print("[dispatch:array] skipping non-object channel entry", flush=True)
                continue
            tag = entry.get("channel")
            spec = channel_map.get(tag)
            if spec is None:
                print(f"[dispatch:array] skipping unknown channel tag {tag!r}", flush=True)
                continue

            msg_rate = entry.get("sampling_rate")
            if isinstance(msg_rate, int) and msg_rate != spec.expected_rate:
                print(
                    f"[dispatch:{spec.role}] sampling_rate mismatch: "
                    f"message={msg_rate}, expected={spec.expected_rate} (continuing)",
                    flush=True,
                )

            readings = entry.get("readings")
            if not isinstance(readings, list):
                print(
                    f"[dispatch:{spec.role}] skipping: 'readings' missing or not a list",
                    flush=True,
                )
                continue

            try:
                buffer.load_buffer(spec.role, ts, readings)
            except Exception as exc:
                print(f"[dispatch:{spec.role}] buffer error: {exc!r}", flush=True)
                raise

    return listener
