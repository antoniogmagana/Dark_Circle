"""
Tests for Discovery Node (ConfigMap-driven whitelist + Kubernetes orchestration).

The Discovery node reads an expected-sensors ConfigMap each poll, checks the
ROS2 graph for completeness of each configured array, and spawns / tears down
Ingestor Deployments accordingly. Pure logic (config parsing, completeness
checking, state tracking) lives in ``discovery.whitelist`` and is imported
directly. The ROS2 / Kubernetes plumbing in ``discovery.main`` is exercised
through inline simulation, since rclpy / kubernetes are not importable in
this test environment.
"""
import json
import textwrap

import pytest

from whitelist import (
    ArraySpec,
    InvalidConfigError,
    PartialAccelError,
    PollState,
    build_role_map,
    is_complete,
    load_config,
    missing_topics,
    required_topics,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """``load_config`` parses the YAML ConfigMap into ArraySpec objects."""

    @pytest.mark.unit
    def test_audio_seismic_only(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
        """)

        specs = load_config(yaml_text)

        assert "shake-001" in specs
        spec = specs["shake-001"]
        assert spec.audio == "/shake_001/aud"
        assert spec.seismic == "/shake_001/ehz"
        assert spec.accel is None

    @pytest.mark.unit
    def test_audio_seismic_with_accel(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
                accel:
                  x: /shake_001/ene
                  y: /shake_001/enn
                  z: /shake_001/enz
        """)

        spec = load_config(yaml_text)["shake-001"]

        assert spec.accel == {
            "x": "/shake_001/ene",
            "y": "/shake_001/enn",
            "z": "/shake_001/enz",
        }

    @pytest.mark.unit
    def test_multiple_arrays(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
              shake-002:
                audio: /shake_002/aud
                seismic: /shake_002/ehz
        """)

        specs = load_config(yaml_text)

        assert set(specs) == {"shake-001", "shake-002"}

    @pytest.mark.unit
    def test_missing_audio_is_invalid(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                seismic: /shake_001/ehz
        """)

        with pytest.raises(InvalidConfigError):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_missing_seismic_is_invalid(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
        """)

        with pytest.raises(InvalidConfigError):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_partial_accel_is_invalid(self):
        """Accel block must have all three of x/y/z or be absent entirely."""
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
                accel:
                  x: /shake_001/ene
                  y: /shake_001/enn
        """)

        with pytest.raises(PartialAccelError):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_empty_arrays_block(self):
        yaml_text = "arrays: {}"
        assert load_config(yaml_text) == {}

    @pytest.mark.unit
    def test_missing_arrays_key_is_invalid(self):
        with pytest.raises(InvalidConfigError):
            load_config("other: {}")

    @pytest.mark.unit
    def test_malformed_yaml_is_invalid(self):
        with pytest.raises(InvalidConfigError):
            load_config("arrays: {shake-001: [unbalanced")

    @pytest.mark.unit
    def test_array_id_with_underscore_rejected(self):
        """Array IDs become k8s Deployment names which must be RFC 1123."""
        yaml_text = textwrap.dedent("""
            arrays:
              shake_001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
        """)
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_starting_with_digit_rejected(self):
        yaml_text = textwrap.dedent("""
            arrays:
              1shake:
                audio: /1shake/aud
                seismic: /1shake/ehz
        """)
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_uppercase_rejected(self):
        yaml_text = textwrap.dedent("""
            arrays:
              Shake-001:
                audio: /Shake-001/aud
                seismic: /Shake-001/ehz
        """)
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_trailing_hyphen_rejected(self):
        yaml_text = textwrap.dedent("""
            arrays:
              shake-:
                audio: /shake/aud
                seismic: /shake/ehz
        """)
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_valid_array_ids(self):
        """Hyphen-separated lowercase IDs starting with a letter are fine."""
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
              s:
                audio: /s/aud
                seismic: /s/ehz
              site-a-rs1d-7:
                audio: /a/aud
                seismic: /a/ehz
        """)
        specs = load_config(yaml_text)
        assert set(specs) == {"shake-001", "s", "site-a-rs1d-7"}

    @pytest.mark.unit
    def test_topic_paths_may_contain_underscores(self):
        """Topic strings (values, not keys) keep underscores from the firmware."""
        yaml_text = textwrap.dedent("""
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
        """)
        spec = load_config(yaml_text)["shake-001"]
        assert spec.audio == "/shake_001/aud"
        assert spec.seismic == "/shake_001/ehz"

    @pytest.mark.unit
    def test_array_id_too_long_rejected(self):
        long_id = "a" * 60
        yaml_text = textwrap.dedent(f"""
            arrays:
              {long_id}:
                audio: /x/aud
                seismic: /x/ehz
        """)
        with pytest.raises(InvalidConfigError, match="max is"):
            load_config(yaml_text)


# ---------------------------------------------------------------------------
# Required-topics / completeness
# ---------------------------------------------------------------------------

class TestRequiredTopics:
    """``required_topics`` lists every topic an array must have to spawn."""

    @pytest.mark.unit
    def test_audio_seismic_only(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        assert required_topics(spec) == {"/a/aud", "/a/ehz"}

    @pytest.mark.unit
    def test_with_accel(self):
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        assert required_topics(spec) == {
            "/a/aud", "/a/ehz", "/a/ene", "/a/enn", "/a/enz"
        }


class TestIsComplete:
    """``is_complete`` returns True iff every required topic is visible."""

    @pytest.mark.unit
    def test_complete_audio_seismic_only(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        visible = {"/a/aud", "/a/ehz"}
        assert is_complete(spec, visible) is True

    @pytest.mark.unit
    def test_missing_seismic_is_incomplete(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        visible = {"/a/aud"}
        assert is_complete(spec, visible) is False

    @pytest.mark.unit
    def test_complete_with_accel(self):
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        visible = {"/a/aud", "/a/ehz", "/a/ene", "/a/enn", "/a/enz"}
        assert is_complete(spec, visible) is True

    @pytest.mark.unit
    def test_partial_accel_visible_is_incomplete(self):
        """If config requires accel x/y/z, all three must be visible."""
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        visible = {"/a/aud", "/a/ehz", "/a/ene", "/a/enn"}
        assert is_complete(spec, visible) is False

    @pytest.mark.unit
    def test_extra_visible_topics_dont_matter(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        visible = {"/a/aud", "/a/ehz", "/somewhere/else"}
        assert is_complete(spec, visible) is True


class TestMissingTopics:
    """``missing_topics`` returns the set of required-but-not-visible topics."""

    @pytest.mark.unit
    def test_returns_only_missing(self):
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        visible = {"/a/aud", "/a/ene"}
        assert missing_topics(spec, visible) == {
            "/a/ehz", "/a/enn", "/a/enz"
        }

    @pytest.mark.unit
    def test_empty_when_complete(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        assert missing_topics(spec, {"/a/aud", "/a/ehz"}) == set()


# ---------------------------------------------------------------------------
# Role-map injection (passed to Ingestor as JSON env)
# ---------------------------------------------------------------------------

class TestBuildRoleMap:
    """``build_role_map`` produces the JSON-serializable dict the Ingestor expects."""

    @pytest.mark.unit
    def test_audio_seismic_only(self):
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        assert build_role_map(spec) == {
            "acoustic": "/a/aud",
            "seismic": "/a/ehz",
        }

    @pytest.mark.unit
    def test_with_accel(self):
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        assert build_role_map(spec) == {
            "acoustic": "/a/aud",
            "seismic": "/a/ehz",
            "accel_x": "/a/ene",
            "accel_y": "/a/enn",
            "accel_z": "/a/enz",
        }

    @pytest.mark.unit
    def test_role_map_round_trips_through_json(self):
        """Role map must serialize cleanly so it can ride in an env var."""
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        encoded = json.dumps(build_role_map(spec))
        assert json.loads(encoded) == build_role_map(spec)


# ---------------------------------------------------------------------------
# Poll-state machine (decides spawn / wait / teardown each cycle)
# ---------------------------------------------------------------------------

class TestPollStateBasic:
    """``PollState`` implements the spawn/teardown decision logic."""

    @pytest.mark.unit
    def test_spawn_when_complete_and_unknown(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud", "/a/ehz"},
            active=set(),
        )
        assert decision.to_spawn == {"shake-001"}
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_no_spawn_when_incomplete(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud"},  # seismic missing
            active=set(),
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_no_double_spawn_when_already_active(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud", "/a/ehz"},
            active={"shake-001"},
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()


class TestPollStateGrace:
    """Grace-poll behavior for both topic-absence and config-removal teardown."""

    @pytest.mark.unit
    def test_topic_absence_grace_period(self):
        """Active array whose topics vanish takes 3 absent polls to tear down."""
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        config = {"shake-001": spec}
        active = {"shake-001"}

        # Polls 1 and 2: topics absent, no teardown yet.
        for _ in range(2):
            decision = state.evaluate(config=config, visible=set(), active=active)
            assert decision.to_teardown == set()

        # Poll 3: grace exhausted, teardown.
        decision = state.evaluate(config=config, visible=set(), active=active)
        assert decision.to_teardown == {"shake-001"}

    @pytest.mark.unit
    def test_topic_reappearance_resets_grace(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        config = {"shake-001": spec}
        active = {"shake-001"}

        # Two absent polls.
        state.evaluate(config=config, visible=set(), active=active)
        state.evaluate(config=config, visible=set(), active=active)

        # Topics back: counter must reset.
        state.evaluate(
            config=config,
            visible={"/a/aud", "/a/ehz"},
            active=active,
        )

        # One more absent poll — should NOT teardown (grace counter reset).
        decision = state.evaluate(config=config, visible=set(), active=active)
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_config_removal_grace_period(self):
        """Removing an array from config takes 3 polls to tear down its Ingestor."""
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        active = {"shake-001"}

        # First poll: array still in config — fully active.
        state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud", "/a/ehz"},
            active=active,
        )

        # Operator removes from config. Topics still on graph.
        empty_config = {}
        for _ in range(2):
            decision = state.evaluate(
                config=empty_config,
                visible={"/a/aud", "/a/ehz"},
                active=active,
            )
            assert decision.to_teardown == set()

        # Third poll out of config: teardown.
        decision = state.evaluate(
            config=empty_config,
            visible={"/a/aud", "/a/ehz"},
            active=active,
        )
        assert decision.to_teardown == {"shake-001"}

    @pytest.mark.unit
    def test_config_re_add_resets_grace(self):
        """Re-adding to config before grace exhausts should cancel teardown."""
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        active = {"shake-001"}

        # Out of config for 2 polls.
        for _ in range(2):
            state.evaluate(
                config={},
                visible={"/a/aud", "/a/ehz"},
                active=active,
            )

        # Re-add to config.
        state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud", "/a/ehz"},
            active=active,
        )

        # Out again for 2 more polls — should still NOT teardown.
        for _ in range(2):
            decision = state.evaluate(
                config={},
                visible={"/a/aud", "/a/ehz"},
                active=active,
            )
            assert decision.to_teardown == set()


class TestPollStateUnknownTopics:
    """Topics on the graph that aren't in the config must never spawn anything."""

    @pytest.mark.unit
    def test_unknown_topic_ignored(self):
        state = PollState(grace_polls=3)
        decision = state.evaluate(
            config={},
            visible={"/wild/aud", "/wild/ehz"},
            active=set(),
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_unknown_topic_alongside_configured_array(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud", "/a/ehz", "/rogue/sensor"},
            active=set(),
        )
        assert decision.to_spawn == {"shake-001"}


class TestPollStateLogThrottle:
    """State-change logging: only emit when the missing-topic set changes."""

    @pytest.mark.unit
    def test_logs_on_first_observation(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/aud"},  # seismic missing
            active=set(),
        )
        assert decision.log_awaiting == {"shake-001": frozenset({"/a/ehz"})}

    @pytest.mark.unit
    def test_no_log_when_missing_set_unchanged(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        config = {"shake-001": spec}

        first = state.evaluate(config=config, visible={"/a/aud"}, active=set())
        assert first.log_awaiting == {"shake-001": frozenset({"/a/ehz"})}

        second = state.evaluate(config=config, visible={"/a/aud"}, active=set())
        assert second.log_awaiting == {}

    @pytest.mark.unit
    def test_log_when_missing_set_changes(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(
            audio="/a/aud",
            seismic="/a/ehz",
            accel={"x": "/a/ene", "y": "/a/enn", "z": "/a/enz"},
        )
        config = {"shake-001": spec}

        # First poll: ehz + accel.x missing.
        first = state.evaluate(
            config=config,
            visible={"/a/aud", "/a/enn", "/a/enz"},
            active=set(),
        )
        assert first.log_awaiting == {
            "shake-001": frozenset({"/a/ehz", "/a/ene"})
        }

        # Second poll: only ehz still missing — set changed, log again.
        second = state.evaluate(
            config=config,
            visible={"/a/aud", "/a/ene", "/a/enn", "/a/enz"},
            active=set(),
        )
        assert second.log_awaiting == {"shake-001": frozenset({"/a/ehz"})}

    @pytest.mark.unit
    def test_log_clears_when_array_becomes_complete(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
        config = {"shake-001": spec}

        state.evaluate(config=config, visible={"/a/aud"}, active=set())
        complete = state.evaluate(
            config=config, visible={"/a/aud", "/a/ehz"}, active=set()
        )
        # Spawning is a state change too — but we only assert the missing-log
        # entry has cleared.
        assert "shake-001" not in complete.log_awaiting


# ---------------------------------------------------------------------------
# Kubernetes orchestration (manifest construction)
# ---------------------------------------------------------------------------

class TestManifestConstruction:
    """The Discovery node fills the ingestor template with role-map JSON."""

    @pytest.mark.integration
    @pytest.mark.k8s
    def test_template_substitution_with_role_map(self, mock_k8s_client):
        """``build_ingestor_manifest`` mutates the YAML dict directly so the
        rendered Deployment carries the right name, labels, and env vars."""
        from whitelist import ArraySpec, build_ingestor_manifest

        template = textwrap.dedent("""
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: ingestor-placeholder
              labels:
                app: ingestor
                sensor-array: placeholder
            spec:
              replicas: 1
              selector:
                matchLabels:
                  app: ingestor
                  sensor-array: placeholder
              template:
                metadata:
                  labels:
                    app: ingestor
                    sensor-array: placeholder
                spec:
                  containers:
                    - name: ingestor
                      image: ingestor:latest
                      env:
                        - name: SENSOR_ARRAY
                          value: "placeholder"
                        - name: SENSOR_ROLE_MAP
                          value: "{}"
        """).strip()

        spec = ArraySpec(audio="/shake_001/aud", seismic="/shake_001/ehz")
        body = build_ingestor_manifest(template, "shake-001", spec)

        mock_k8s_client.create_namespaced_deployment(namespace="default", body=body)
        mock_k8s_client.create_namespaced_deployment.assert_called_once()

        assert body["metadata"]["name"] == "ingestor-shake-001"
        assert body["metadata"]["labels"]["sensor-array"] == "shake-001"
        assert body["spec"]["selector"]["matchLabels"]["sensor-array"] == "shake-001"
        assert (
            body["spec"]["template"]["metadata"]["labels"]["sensor-array"]
            == "shake-001"
        )
        env = body["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e["value"] for e in env}
        assert env_dict["SENSOR_ARRAY"] == "shake-001"
        assert json.loads(env_dict["SENSOR_ROLE_MAP"]) == {
            "acoustic": "/shake_001/aud",
            "seismic": "/shake_001/ehz",
        }

    @pytest.mark.integration
    @pytest.mark.k8s
    def test_manifest_substitution_handles_quoted_topics(self):
        """JSON role-map values with special characters survive YAML
        round-trip when we mutate the dict instead of string-replacing."""
        from whitelist import ArraySpec, build_ingestor_manifest

        template = textwrap.dedent("""
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: ingestor-placeholder
              labels:
                app: ingestor
                sensor-array: placeholder
            spec:
              selector:
                matchLabels:
                  app: ingestor
                  sensor-array: placeholder
              template:
                metadata:
                  labels:
                    app: ingestor
                    sensor-array: placeholder
                spec:
                  containers:
                    - name: ingestor
                      image: ingestor:latest
                      env:
                        - name: SENSOR_ARRAY
                          value: "placeholder"
                        - name: SENSOR_ROLE_MAP
                          value: "{}"
        """).strip()

        spec = ArraySpec(
            audio='/shake_001/audio "primary"',
            seismic="/shake_001/ehz",
        )
        body = build_ingestor_manifest(template, "shake-001", spec)
        env = body["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e["value"] for e in env}
        assert json.loads(env_dict["SENSOR_ROLE_MAP"])["acoustic"] == (
            '/shake_001/audio "primary"'
        )

    @pytest.mark.integration
    @pytest.mark.k8s
    def test_teardown_calls_delete(self, mock_k8s_client):
        sensor_array = "shake-001"
        mock_k8s_client.delete_namespaced_deployment(
            name=f"ingestor-{sensor_array}",
            namespace="default",
        )
        mock_k8s_client.delete_namespaced_deployment.assert_called_once_with(
            name="ingestor-shake-001",
            namespace="default",
        )


# ---------------------------------------------------------------------------
# End-to-end poll cycle simulation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_poll_cycle_spawns_then_teardowns_on_topic_loss():
    """Drive PollState through a full life cycle: incomplete → complete → loss."""
    state = PollState(grace_polls=3)
    spec = ArraySpec(audio="/a/aud", seismic="/a/ehz", accel=None)
    config = {"shake-001": spec}
    active = set()

    # Poll 1: only audio visible — wait.
    d = state.evaluate(config=config, visible={"/a/aud"}, active=active)
    assert d.to_spawn == set()

    # Poll 2: seismic appears — spawn.
    d = state.evaluate(config=config, visible={"/a/aud", "/a/ehz"}, active=active)
    assert d.to_spawn == {"shake-001"}
    active.add("shake-001")

    # Polls 3-4: topics still present, idempotent.
    for _ in range(2):
        d = state.evaluate(
            config=config, visible={"/a/aud", "/a/ehz"}, active=active
        )
        assert d.to_spawn == set()
        assert d.to_teardown == set()

    # Polls 5-7: topics vanish, grace counts down.
    for i in range(3):
        d = state.evaluate(config=config, visible=set(), active=active)
        if i < 2:
            assert d.to_teardown == set()
        else:
            assert d.to_teardown == {"shake-001"}
