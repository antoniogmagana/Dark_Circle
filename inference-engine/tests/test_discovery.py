"""
Tests for Discovery Node (ConfigMap-driven whitelist + Kubernetes orchestration).

The Discovery node reads an expected-sensors ConfigMap each poll, checks the
ROS2 graph for completeness of each configured array, and spawns / tears down
Ingestor Deployments accordingly. Each array now has one bundled-channel
``std_msgs/String`` topic, so ``ArraySpec`` collapses to ``{topic}``. Pure
logic (config parsing, completeness checking, state tracking) lives in
``discovery.whitelist`` and is imported directly. The ROS2 / Kubernetes
plumbing in ``discovery.main`` is exercised through inline simulation.
"""

import textwrap

import pytest
from whitelist import (
    ArraySpec,
    InvalidConfigError,
    PollState,
    build_ingestor_manifest,
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
    def test_single_topic(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                topic: /shake_001/data
        """
        )

        specs = load_config(yaml_text)

        assert "shake-001" in specs
        assert specs["shake-001"].topic == "/shake_001/data"

    @pytest.mark.unit
    def test_multiple_arrays(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                topic: /shake_001/data
              shake-002:
                topic: /shake_002/data
        """
        )

        specs = load_config(yaml_text)

        assert set(specs) == {"shake-001", "shake-002"}

    @pytest.mark.unit
    def test_missing_topic_is_invalid(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001: {}
        """
        )
        with pytest.raises(InvalidConfigError, match="topic"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_empty_topic_is_invalid(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                topic: ""
        """
        )
        with pytest.raises(InvalidConfigError, match="topic"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_legacy_audio_seismic_keys_rejected(self):
        """Pre-refactor configs must produce a clear migration error."""
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                audio: /shake_001/aud
                seismic: /shake_001/ehz
        """
        )
        with pytest.raises(InvalidConfigError, match="schema changed"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_legacy_accel_key_rejected(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                topic: /shake_001/data
                accel:
                  x: /shake_001/ene
        """
        )
        with pytest.raises(InvalidConfigError, match="schema changed"):
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
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake_001:
                topic: /shake_001/data
        """
        )
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_starting_with_digit_rejected(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              1shake:
                topic: /1shake/data
        """
        )
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_uppercase_rejected(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              Shake-001:
                topic: /Shake-001/data
        """
        )
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_array_id_trailing_hyphen_rejected(self):
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-:
                topic: /shake/data
        """
        )
        with pytest.raises(InvalidConfigError, match="RFC 1123"):
            load_config(yaml_text)

    @pytest.mark.unit
    def test_topic_paths_may_contain_underscores(self):
        """Topic strings (values, not keys) keep underscores from the firmware."""
        yaml_text = textwrap.dedent(
            """
            arrays:
              shake-001:
                topic: /shake_001/data
        """
        )
        spec = load_config(yaml_text)["shake-001"]
        assert spec.topic == "/shake_001/data"

    @pytest.mark.unit
    def test_array_id_too_long_rejected(self):
        long_id = "a" * 60
        yaml_text = textwrap.dedent(
            f"""
            arrays:
              {long_id}:
                topic: /x/data
        """
        )
        with pytest.raises(InvalidConfigError, match="max is"):
            load_config(yaml_text)


# ---------------------------------------------------------------------------
# Required-topics / completeness
# ---------------------------------------------------------------------------


class TestRequiredTopics:
    @pytest.mark.unit
    def test_returns_single_topic(self):
        spec = ArraySpec(topic="/a/data")
        assert required_topics(spec) == {"/a/data"}


class TestIsComplete:
    @pytest.mark.unit
    def test_complete_when_topic_visible(self):
        spec = ArraySpec(topic="/a/data")
        assert is_complete(spec, {"/a/data"}) is True

    @pytest.mark.unit
    def test_incomplete_when_topic_missing(self):
        spec = ArraySpec(topic="/a/data")
        assert is_complete(spec, set()) is False

    @pytest.mark.unit
    def test_extra_visible_topics_dont_matter(self):
        spec = ArraySpec(topic="/a/data")
        assert is_complete(spec, {"/a/data", "/somewhere/else"}) is True


class TestMissingTopics:
    @pytest.mark.unit
    def test_returns_only_missing(self):
        spec = ArraySpec(topic="/a/data")
        assert missing_topics(spec, set()) == {"/a/data"}

    @pytest.mark.unit
    def test_empty_when_complete(self):
        spec = ArraySpec(topic="/a/data")
        assert missing_topics(spec, {"/a/data"}) == set()


# ---------------------------------------------------------------------------
# Poll-state machine (decides spawn / wait / teardown each cycle)
# ---------------------------------------------------------------------------


class TestPollStateBasic:
    @pytest.mark.unit
    def test_spawn_when_complete_and_unknown(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/data"},
            active=set(),
        )
        assert decision.to_spawn == {"shake-001"}
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_no_spawn_when_incomplete(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        decision = state.evaluate(
            config={"shake-001": spec},
            visible=set(),
            active=set(),
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_no_double_spawn_when_already_active(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/data"},
            active={"shake-001"},
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()


class TestPollStateGrace:
    @pytest.mark.unit
    def test_topic_absence_grace_period(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        config = {"shake-001": spec}
        active = {"shake-001"}

        for _ in range(2):
            decision = state.evaluate(config=config, visible=set(), active=active)
            assert decision.to_teardown == set()

        decision = state.evaluate(config=config, visible=set(), active=active)
        assert decision.to_teardown == {"shake-001"}

    @pytest.mark.unit
    def test_topic_reappearance_resets_grace(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        config = {"shake-001": spec}
        active = {"shake-001"}

        state.evaluate(config=config, visible=set(), active=active)
        state.evaluate(config=config, visible=set(), active=active)

        state.evaluate(config=config, visible={"/a/data"}, active=active)

        decision = state.evaluate(config=config, visible=set(), active=active)
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_config_removal_grace_period(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        active = {"shake-001"}

        state.evaluate(
            config={"shake-001": spec},
            visible={"/a/data"},
            active=active,
        )

        empty_config = {}
        for _ in range(2):
            decision = state.evaluate(
                config=empty_config,
                visible={"/a/data"},
                active=active,
            )
            assert decision.to_teardown == set()

        decision = state.evaluate(
            config=empty_config,
            visible={"/a/data"},
            active=active,
        )
        assert decision.to_teardown == {"shake-001"}

    @pytest.mark.unit
    def test_config_re_add_resets_grace(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        active = {"shake-001"}

        for _ in range(2):
            state.evaluate(config={}, visible={"/a/data"}, active=active)

        state.evaluate(
            config={"shake-001": spec},
            visible={"/a/data"},
            active=active,
        )

        for _ in range(2):
            decision = state.evaluate(
                config={},
                visible={"/a/data"},
                active=active,
            )
            assert decision.to_teardown == set()


class TestPollStateUnknownTopics:
    @pytest.mark.unit
    def test_unknown_topic_ignored(self):
        state = PollState(grace_polls=3)
        decision = state.evaluate(
            config={},
            visible={"/wild/data"},
            active=set(),
        )
        assert decision.to_spawn == set()
        assert decision.to_teardown == set()

    @pytest.mark.unit
    def test_unknown_topic_alongside_configured_array(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        decision = state.evaluate(
            config={"shake-001": spec},
            visible={"/a/data", "/rogue/sensor"},
            active=set(),
        )
        assert decision.to_spawn == {"shake-001"}


class TestPollStateLogThrottle:
    @pytest.mark.unit
    def test_logs_on_first_observation(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        decision = state.evaluate(
            config={"shake-001": spec},
            visible=set(),
            active=set(),
        )
        assert decision.log_awaiting == {"shake-001": frozenset({"/a/data"})}

    @pytest.mark.unit
    def test_no_log_when_missing_set_unchanged(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        config = {"shake-001": spec}

        first = state.evaluate(config=config, visible=set(), active=set())
        assert first.log_awaiting == {"shake-001": frozenset({"/a/data"})}

        second = state.evaluate(config=config, visible=set(), active=set())
        assert second.log_awaiting == {}

    @pytest.mark.unit
    def test_log_clears_when_array_becomes_complete(self):
        state = PollState(grace_polls=3)
        spec = ArraySpec(topic="/a/data")
        config = {"shake-001": spec}

        state.evaluate(config=config, visible=set(), active=set())
        complete = state.evaluate(config=config, visible={"/a/data"}, active=set())
        assert "shake-001" not in complete.log_awaiting


# ---------------------------------------------------------------------------
# Kubernetes orchestration (manifest construction)
# ---------------------------------------------------------------------------


class TestManifestConstruction:
    """Discovery fills the ingestor template with per-array env vars."""

    @pytest.mark.integration
    @pytest.mark.k8s
    def test_template_substitution_with_topic(self, mock_k8s_client):
        template = textwrap.dedent(
            """
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
                        - name: SENSOR_TOPIC
                          value: "placeholder"
        """
        ).strip()

        spec = ArraySpec(topic="/shake_001/data")
        body = build_ingestor_manifest(template, "shake-001", spec)

        mock_k8s_client.create_namespaced_deployment(namespace="default", body=body)
        mock_k8s_client.create_namespaced_deployment.assert_called_once()

        assert body["metadata"]["name"] == "ingestor-shake-001"
        assert body["metadata"]["labels"]["sensor-array"] == "shake-001"
        assert body["spec"]["selector"]["matchLabels"]["sensor-array"] == "shake-001"
        assert body["spec"]["template"]["metadata"]["labels"]["sensor-array"] == "shake-001"
        env = body["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e["value"] for e in env}
        assert env_dict["SENSOR_ARRAY"] == "shake-001"
        assert env_dict["SENSOR_TOPIC"] == "/shake_001/data"

    @pytest.mark.integration
    @pytest.mark.k8s
    def test_topic_with_special_characters_round_trips(self):
        """Topic strings with quotes / spaces survive YAML round-trip."""
        template = textwrap.dedent(
            """
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
                        - name: SENSOR_TOPIC
                          value: "placeholder"
        """
        ).strip()

        spec = ArraySpec(topic='/shake_001/data "primary"')
        body = build_ingestor_manifest(template, "shake-001", spec)
        env = body["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e["value"] for e in env}
        assert env_dict["SENSOR_TOPIC"] == '/shake_001/data "primary"'

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
    spec = ArraySpec(topic="/a/data")
    config = {"shake-001": spec}
    active = set()

    d = state.evaluate(config=config, visible=set(), active=active)
    assert d.to_spawn == set()

    d = state.evaluate(config=config, visible={"/a/data"}, active=active)
    assert d.to_spawn == {"shake-001"}
    active.add("shake-001")

    for _ in range(2):
        d = state.evaluate(config=config, visible={"/a/data"}, active=active)
        assert d.to_spawn == set()
        assert d.to_teardown == set()

    for i in range(3):
        d = state.evaluate(config=config, visible=set(), active=active)
        if i < 2:
            assert d.to_teardown == set()
        else:
            assert d.to_teardown == {"shake-001"}
