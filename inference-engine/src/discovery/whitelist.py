"""
Pure logic for the ConfigMap-driven sensor whitelist.

This module is intentionally free of rclpy / kubernetes imports so it can be
unit-tested without the ROS2 or Kubernetes runtime. ``discovery.main``
imports from here and adds the ROS2 graph poll + Kubernetes API plumbing on
top.
"""
import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import yaml

# Array IDs become Kubernetes Deployment names (``ingestor-<array_id>``),
# which must be RFC 1123 subdomains: lowercase alphanumeric + hyphens, start
# with a letter, end alphanumeric, ≤63 chars after the ``ingestor-`` prefix.
RFC_1123_ARRAY_ID = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
MAX_ARRAY_ID_LEN = 63 - len("ingestor-")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class InvalidConfigError(ValueError):
    """The ConfigMap YAML is malformed or missing required keys."""


class PartialAccelError(InvalidConfigError):
    """An ``accel`` block is present but doesn't define all of x, y, z."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArraySpec:
    """One configured sensor array.

    ``accel`` is None when the operator chose not to include accelerometer
    data; when present it must contain all three of x/y/z.
    """
    audio: str
    seismic: str
    accel: dict | None = None


@dataclass
class PollDecision:
    """The action set the manager should apply this tick."""
    to_spawn: set = field(default_factory=set)
    to_teardown: set = field(default_factory=set)
    log_awaiting: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(yaml_text: str) -> dict[str, ArraySpec]:
    """Parse the expected-sensors ConfigMap into ``{array_id: ArraySpec}``."""
    try:
        doc = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise InvalidConfigError(f"malformed YAML: {exc}") from exc

    if not isinstance(doc, dict) or "arrays" not in doc:
        raise InvalidConfigError("config must define a top-level 'arrays' key")

    arrays = doc["arrays"] or {}
    if not isinstance(arrays, dict):
        raise InvalidConfigError("'arrays' must be a mapping")

    specs: dict[str, ArraySpec] = {}
    for array_id, entry in arrays.items():
        if not isinstance(array_id, str) or not RFC_1123_ARRAY_ID.match(array_id):
            raise InvalidConfigError(
                f"invalid array id {array_id!r}: must be lowercase RFC 1123 "
                "subdomain (letters/digits/hyphens, start with letter, end "
                "alphanumeric); underscores are not allowed because the id "
                "becomes a Kubernetes Deployment name"
            )
        if len(array_id) > MAX_ARRAY_ID_LEN:
            raise InvalidConfigError(
                f"array id {array_id!r} is {len(array_id)} chars; max is "
                f"{MAX_ARRAY_ID_LEN} so ``ingestor-<id>`` fits in 63 chars"
            )
        if not isinstance(entry, dict):
            raise InvalidConfigError(f"{array_id}: entry must be a mapping")

        audio = entry.get("audio")
        seismic = entry.get("seismic")
        if not audio:
            raise InvalidConfigError(f"{array_id}: 'audio' is required")
        if not seismic:
            raise InvalidConfigError(f"{array_id}: 'seismic' is required")

        accel = entry.get("accel")
        if accel is not None:
            if not isinstance(accel, dict):
                raise InvalidConfigError(
                    f"{array_id}: 'accel' must be a mapping with x/y/z"
                )
            missing = {"x", "y", "z"} - set(accel)
            if missing:
                raise PartialAccelError(
                    f"{array_id}: accel block missing keys {sorted(missing)}; "
                    "accel must define all of x, y, z or be omitted entirely"
                )

        specs[array_id] = ArraySpec(audio=audio, seismic=seismic, accel=accel)

    return specs


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------

def required_topics(spec: ArraySpec) -> set[str]:
    """Topics that must be on the ROS2 graph for ``spec`` to spawn."""
    topics = {spec.audio, spec.seismic}
    if spec.accel is not None:
        topics.update(spec.accel.values())
    return topics


def is_complete(spec: ArraySpec, visible: Iterable[str]) -> bool:
    return required_topics(spec).issubset(set(visible))


def missing_topics(spec: ArraySpec, visible: Iterable[str]) -> set[str]:
    return required_topics(spec) - set(visible)


# ---------------------------------------------------------------------------
# Role-map injection (env var sent to the Ingestor)
# ---------------------------------------------------------------------------

def build_role_map(spec: ArraySpec) -> dict[str, str]:
    """``{role: topic}`` mapping consumed by the Ingestor at startup."""
    role_map = {
        "acoustic": spec.audio,
        "seismic": spec.seismic,
    }
    if spec.accel is not None:
        role_map["accel_x"] = spec.accel["x"]
        role_map["accel_y"] = spec.accel["y"]
        role_map["accel_z"] = spec.accel["z"]
    return role_map


# ---------------------------------------------------------------------------
# Ingestor manifest construction
# ---------------------------------------------------------------------------

def build_ingestor_manifest(
    template_yaml: str, sensor_array: str, spec: ArraySpec,
) -> dict:
    """Render the ingestor Deployment for ``sensor_array``.

    Loads the template YAML once, then mutates the dict directly so JSON
    role-map values containing quotes / braces never collide with the
    placeholder string. Returns a dict ready for the Kubernetes API.
    """
    body = copy.deepcopy(yaml.safe_load(template_yaml))
    role_map_json = json.dumps(build_role_map(spec))
    deployment_name = f"ingestor-{sensor_array}"

    body["metadata"]["name"] = deployment_name
    body["metadata"].setdefault("labels", {})
    body["metadata"]["labels"]["sensor-array"] = sensor_array

    selector_labels = body["spec"]["selector"]["matchLabels"]
    selector_labels["sensor-array"] = sensor_array

    pod_meta = body["spec"]["template"]["metadata"]
    pod_meta.setdefault("labels", {})
    pod_meta["labels"]["sensor-array"] = sensor_array

    for container in body["spec"]["template"]["spec"]["containers"]:
        env = container.setdefault("env", [])
        _set_env(env, "SENSOR_ARRAY", sensor_array)
        _set_env(env, "SENSOR_ROLE_MAP", role_map_json)

    return body


def _set_env(env_list: list, name: str, value: str) -> None:
    """Replace or append a name/value pair in a Kubernetes env list."""
    for entry in env_list:
        if entry.get("name") == name:
            entry["value"] = value
            entry.pop("valueFrom", None)
            return
    env_list.append({"name": name, "value": value})


# ---------------------------------------------------------------------------
# Poll state machine
# ---------------------------------------------------------------------------

class PollState:
    """Stateful decision engine for one Discovery node.

    ``evaluate`` is called once per poll tick. The caller supplies the latest
    config (parsed from the ConfigMap), the set of currently-visible sensor
    topics on the ROS2 graph, and the set of currently-active Ingestor
    deployments. The returned ``PollDecision`` lists which arrays to spawn,
    which to tear down, and which incomplete arrays should generate an
    ``awaiting...`` log line this cycle (state-change throttle).
    """

    def __init__(self, grace_polls: int = 3):
        self.grace_polls = grace_polls
        self._absent_counts: dict[str, int] = defaultdict(int)
        self._last_missing: dict[str, frozenset[str]] = {}

    def evaluate(
        self,
        config: dict[str, ArraySpec],
        visible: set[str],
        active: set[str],
    ) -> PollDecision:
        decision = PollDecision()

        # ----- spawn pass: configured + complete + not yet active --------
        for array_id, spec in config.items():
            if array_id in active:
                continue
            if is_complete(spec, visible):
                decision.to_spawn.add(array_id)
                self._last_missing.pop(array_id, None)
            else:
                missing = frozenset(missing_topics(spec, visible))
                if self._last_missing.get(array_id) != missing:
                    decision.log_awaiting[array_id] = missing
                    self._last_missing[array_id] = missing

        # ----- teardown pass: any active array failing either condition --
        # An array is "healthy" iff (a) still in config AND (b) all required
        # topics still visible. Anything else accumulates absent_counts.
        for array_id in list(active):
            spec = config.get(array_id)
            healthy = spec is not None and is_complete(spec, visible)
            if healthy:
                self._absent_counts.pop(array_id, None)
                continue

            self._absent_counts[array_id] += 1
            if self._absent_counts[array_id] >= self.grace_polls:
                decision.to_teardown.add(array_id)
                self._absent_counts.pop(array_id, None)
                self._last_missing.pop(array_id, None)

        return decision
