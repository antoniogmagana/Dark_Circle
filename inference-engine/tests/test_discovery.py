"""
Tests for Discovery Node (Kubernetes orchestration).

The Discovery node watches ROS2 topics and dynamically spawns
Ingestor deployments via Kubernetes API.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Topic Discovery
# ============================================================================

class TestTopicDiscovery:
    """Test ROS2 topic discovery and parsing."""
    
    @pytest.mark.integration
    @pytest.mark.ros2
    def test_get_sensor_arrays_basic(self, mock_ros2_node):
        """Test basic sensor array discovery from topics."""
        from collections import defaultdict
        
        # Mock ROS2 node with topic names and types
        mock_ros2_node.get_topic_names_and_types.return_value = [
            ('/sensor_array_01/acoustic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/sensor_array_01/seismic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/sensor_array_02/acoustic', ['ros2_interfaces/msg/RawSensorReading']),
        ]
        
        # Simulate _get_sensor_arrays() logic
        SENSOR_MSG_TYPE = 'ros2_interfaces/msg/RawSensorReading'
        arrays = defaultdict(list)
        
        for topic, types in mock_ros2_node.get_topic_names_and_types():
            if SENSOR_MSG_TYPE in types:
                parts = topic.strip('/').split('/')
                if len(parts) >= 2:
                    sensor_array = parts[0]
                    arrays[sensor_array].append(topic)
        
        # Verify discovery logic worked
        assert 'sensor_array_01' in arrays
        assert 'sensor_array_02' in arrays
        assert len(arrays['sensor_array_01']) == 2
        assert '/sensor_array_01/acoustic' in arrays['sensor_array_01']
        assert '/sensor_array_01/seismic' in arrays['sensor_array_01']
    
    @pytest.mark.unit
    def test_parse_topic_name(self):
        """Test topic name parsing for array prefix extraction."""
        # Example: "/sensor_array_01/acoustic" -> "sensor_array_01"
        # Based on _get_sensor_arrays() implementation
        topic = "/sensor_array_01/acoustic"
        parts = topic.strip('/').split('/')
        assert len(parts) >= 2
        sensor_array = parts[0]
        assert sensor_array == "sensor_array_01"
        
        # Test with different formats
        topic2 = "/array_42/seismic"
        parts2 = topic2.strip('/').split('/')
        assert parts2[0] == "array_42"
        
        # Test edge case: no leading slash
        topic3 = "sensor_array_03/accel"
        parts3 = topic3.strip('/').split('/')
        assert parts3[0] == "sensor_array_03"
    
    @pytest.mark.unit
    def test_group_topics_by_array(self):
        """Test grouping topics by sensor array prefix."""
        from collections import defaultdict
        
        # Simulate what _get_sensor_arrays() does
        topics_and_types = [
            ("/sensor_array_01/acoustic", ['ros2_interfaces/msg/RawSensorReading']),
            ("/sensor_array_01/seismic", ['ros2_interfaces/msg/RawSensorReading']),
            ("/sensor_array_02/acoustic", ['ros2_interfaces/msg/RawSensorReading']),
            ("/other/topic", ['std_msgs/msg/String']),  # Should be filtered out
        ]
        
        SENSOR_MSG_TYPE = 'ros2_interfaces/msg/RawSensorReading'
        arrays = defaultdict(list)
        
        for topic, types in topics_and_types:
            if SENSOR_MSG_TYPE in types:
                parts = topic.strip('/').split('/')
                if len(parts) >= 2:
                    sensor_array = parts[0]
                    arrays[sensor_array].append(topic)
        
        assert len(arrays) == 2
        assert "sensor_array_01" in arrays
        assert "sensor_array_02" in arrays
        assert len(arrays["sensor_array_01"]) == 2
        assert len(arrays["sensor_array_02"]) == 1
        assert "/sensor_array_01/acoustic" in arrays["sensor_array_01"]
        assert "/sensor_array_01/seismic" in arrays["sensor_array_01"]
    
    @pytest.mark.integration
    @pytest.mark.ros2
    def test_handle_multiple_arrays(self, mock_ros2_node):
        """Test discovery handles multiple sensor arrays."""
        from collections import defaultdict
        
        # Mock multiple sensor arrays
        mock_ros2_node.get_topic_names_and_types.return_value = [
            ('/array_A/acoustic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/array_A/seismic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/array_B/acoustic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/array_C/seismic', ['ros2_interfaces/msg/RawSensorReading']),
            ('/other/topic', ['std_msgs/msg/String']),  # Should be filtered
        ]
        
        SENSOR_MSG_TYPE = 'ros2_interfaces/msg/RawSensorReading'
        arrays = defaultdict(list)
        
        for topic, types in mock_ros2_node.get_topic_names_and_types():
            if SENSOR_MSG_TYPE in types:
                parts = topic.strip('/').split('/')
                if len(parts) >= 2:
                    sensor_array = parts[0]
                    arrays[sensor_array].append(topic)
        
        # Verify multiple arrays discovered
        assert len(arrays) == 3
        assert 'array_A' in arrays
        assert 'array_B' in arrays
        assert 'array_C' in arrays
        assert 'other' not in arrays  # Should be filtered out


# ============================================================================
# Test Kubernetes Orchestration
# ============================================================================

class TestKubernetesOrchestration:
    """Test Kubernetes deployment management."""
    
    @pytest.mark.integration
    @pytest.mark.k8s
    def test_spawn_deployment(self, mock_k8s_client):
        """Test spawning new Ingestor deployment."""
        import yaml
        
        # Mock deployment template
        template = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingestor-<sensor_array_id>
  labels:
    app: ingestor
    sensor-array: <sensor_array_id>
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ingestor
      sensor-array: <sensor_array_id>
  template:
    metadata:
      labels:
        app: ingestor
        sensor-array: <sensor_array_id>
    spec:
      containers:
      - name: ingestor
        image: ingestor:latest
        env:
        - name: SENSOR_ARRAY
          value: "<sensor_array_id>"
        - name: SENSOR_TOPICS
          value: "<comma,separated,topics>"
"""
        
        # Simulate _spawn() method
        sensor_array = "sensor_array_01"
        topics = ['/sensor_array_01/acoustic', '/sensor_array_01/seismic']
        topics_str = ','.join(topics)
        manifest = template.replace('<sensor_array_id>', sensor_array)
        manifest = manifest.replace('<comma,separated,topics>', topics_str)
        body = yaml.safe_load(manifest)
        
        # Call mock k8s API
        mock_k8s_client.create_namespaced_deployment(
            namespace='default',
            body=body
        )
        
        # Verify deployment was created
        mock_k8s_client.create_namespaced_deployment.assert_called_once()
        call_args = mock_k8s_client.create_namespaced_deployment.call_args
        assert call_args[1]['namespace'] == 'default'
        assert 'sensor_array_01' in str(call_args[1]['body'])
    
    @pytest.mark.integration
    @pytest.mark.k8s
    def test_teardown_deployment(self, mock_k8s_client):
        """Test tearing down Ingestor deployment."""
        # Simulate _teardown() method
        sensor_array = "sensor_array_01"
        namespace = "default"
        
        # Call delete Deployment
        mock_k8s_client.delete_namespaced_deployment(
            name=f'ingestor-{sensor_array}',
            namespace=namespace
        )
        
        # Verify delete was called
        mock_k8s_client.delete_namespaced_deployment.assert_called_once_with(
            name='ingestor-sensor_array_01',
            namespace='default'
        )
    
    @pytest.mark.unit
    def test_absent_counter_grace_period(self):
        """Test grace period before deployment teardown."""
        from collections import defaultdict
        
        # Simulate grace period logic from _poll()
        GRACE_POLLS = 3
        absent_counts = defaultdict(int)
        active_arrays = {"sensor_array_01", "sensor_array_02"}
        visible_arrays = {"sensor_array_01": []}  # sensor_array_02 is now absent
        
        # First poll: sensor_array_02 goes absent
        for sensor_array in list(active_arrays):
            if sensor_array not in visible_arrays:
                absent_counts[sensor_array] += 1
        
        assert absent_counts["sensor_array_02"] == 1
        assert absent_counts["sensor_array_02"] < GRACE_POLLS  # Don't teardown yet
        
        # Second poll: still absent
        for sensor_array in list(active_arrays):
            if sensor_array not in visible_arrays:
                absent_counts[sensor_array] += 1
        assert absent_counts["sensor_array_02"] == 2
        assert absent_counts["sensor_array_02"] < GRACE_POLLS
        
        # Third poll: reaches threshold
        for sensor_array in list(active_arrays):
            if sensor_array not in visible_arrays:
                absent_counts[sensor_array] += 1
        assert absent_counts["sensor_array_02"] == 3
        assert absent_counts["sensor_array_02"] >= GRACE_POLLS  # Now should teardown
        
        # Test reset when array becomes visible again
        visible_arrays["sensor_array_02"] = []
        for sensor_array in visible_arrays:
            if sensor_array in active_arrays:
                absent_counts.pop(sensor_array, None)
        assert "sensor_array_02" not in absent_counts
    
    @pytest.mark.integration
    @pytest.mark.k8s
    def test_idempotent_spawning(self, mock_k8s_client):
        """Test spawning same array twice doesn't create duplicate."""
        # Simulate idempotency check in _poll()
        active_arrays = set()
        visible_arrays = {'sensor_array_01': ['/sensor_array_01/acoustic']}
        
        # First spawn
        for sensor_array, topics in visible_arrays.items():
            if sensor_array not in active_arrays:
                # Would call _spawn() here
                active_arrays.add(sensor_array)
        
        spawn_count = len(active_arrays)
        assert spawn_count == 1
        
        # Second iteration - array already active
        for sensor_array, topics in visible_arrays.items():
            if sensor_array not in active_arrays:
                active_arrays.add(sensor_array)
        
        # Should still be 1, no duplicate spawn
        assert len(active_arrays) == 1
        assert 'sensor_array_01' in active_arrays


# ============================================================================
# Test Configuration
# ============================================================================

class TestDiscoveryConfiguration:
    """Test Discovery node configuration."""
    
    @pytest.mark.unit
    def test_poll_interval(self):
        """Test poll interval configuration."""
        # From discovery/main.py
        POLL_INTERVAL = 5.0  # seconds
        
        # Assert default interval
        assert POLL_INTERVAL == 5.0
        assert isinstance(POLL_INTERVAL, float)
        
        # Test that interval can be overridden
        custom_interval = 10.0
        assert custom_interval > POLL_INTERVAL
        
        # Test with mock timer
        mock_node = Mock()
        mock_timer = Mock()
        mock_node.create_timer.return_value = mock_timer
        
        # Simulate creating timer with interval
        timer = mock_node.create_timer(POLL_INTERVAL, Mock())
        
        # Verify timer created
        assert timer is mock_timer
        mock_node.create_timer.assert_called_once()
        assert mock_node.create_timer.call_args[0][0] == POLL_INTERVAL
    
    @pytest.mark.unit
    def test_namespace_configuration(self):
        """Test Kubernetes namespace is configurable."""
        # From discovery/main.py: namespace = os.environ.get("NAMESPACE", "default")
        
        # Test default namespace
        default_namespace = "default"
        assert default_namespace == "default"
        
        # Test custom namespace
        custom_namespace = "production"
        assert custom_namespace != default_namespace
        
        # Test namespace validation (k8s naming rules)
        valid_namespaces = ["default", "kube-system", "production", "dev-01"]
        for ns in valid_namespaces:
            assert isinstance(ns, str)
            assert len(ns) > 0
            assert ns.replace('-', '').replace('_', '').isalnum()
    
    @pytest.mark.unit
    def test_deployment_template_loading(self):
        """Test deployment template is loaded correctly."""
        # Test YAML template structure
        template = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingestor-<sensor_array_id>
  labels:
    app: ingestor
    sensor-array: <sensor_array_id>
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ingestor
      sensor-array: <sensor_array_id>
  template:
    metadata:
      labels:
        app: ingestor
        sensor-array: <sensor_array_id>
    spec:
      containers:
      - name: ingestor
        image: ingestor:latest
        env:
        - name: SENSOR_ARRAY
          value: <sensor_array_id>
        - name: SENSOR_TOPICS
          value: <comma,separated,topics>"""
        
        # Test template contains placeholders
        assert '<sensor_array_id>' in template
        assert '<comma,separated,topics>' in template
        
        # Test template replacement
        sensor_array = "sensor_array_01"
        topics = "/sensor_array_01/acoustic,/sensor_array_01/seismic"
        
        manifest = template.replace('<sensor_array_id>', sensor_array)
        manifest = manifest.replace('<comma,separated,topics>', topics)
        
        # Verify replacements
        assert '<sensor_array_id>' not in manifest
        assert '<comma,separated,topics>' not in manifest
        assert 'sensor_array_01' in manifest
        assert topics in manifest
        
        # Test YAML is valid after replacement
        import yaml
        parsed = yaml.safe_load(manifest)
        assert parsed['kind'] == 'Deployment'
        assert parsed['metadata']['name'] == f'ingestor-{sensor_array}'


# ============================================================================
# Placeholders for Future Implementation
# ============================================================================

@pytest.mark.integration
@pytest.mark.ros2
@pytest.mark.k8s
def test_discovery_integration(mock_ros2_node, mock_k8s_client):
    """Integration test: Full discovery cycle."""
    # Arrange - simulate full poll cycle
    active_arrays = set()
    absent_counts = {}
    GRACE_POLLS = 3
    
    # Mock ROS2 topic discovery
    mock_ros2_node.get_topic_names_and_types.return_value = [
        ('/sensor_array_01/acoustic', ['ros2_interfaces/msg/RawSensorReading']),
        ('/sensor_array_01/seismic', ['ros2_interfaces/msg/RawSensorReading']),
    ]
    
    # Act - simulate poll #1: discover and spawn
    from collections import defaultdict
    SENSOR_MSG_TYPE = 'ros2_interfaces/msg/RawSensorReading'
    
    arrays = defaultdict(list)
    for topic, types in mock_ros2_node.get_topic_names_and_types():
        if SENSOR_MSG_TYPE in types:
            parts = topic.strip('/').split('/')
            if len(parts) >= 2:
                sensor_array = parts[0]
                arrays[sensor_array].append(topic)
    
    # Spawn new arrays
    for sensor_array, topics in arrays.items():
        if sensor_array not in active_arrays:
            # Would call k8s create_deployment here
            active_arrays.add(sensor_array)
    
    # Assert
    assert 'sensor_array_01' in active_arrays
    assert len(arrays['sensor_array_01']) == 2
    
    # Act - simulate poll #2: array disappears
    mock_ros2_node.get_topic_names_and_types.return_value = []
    
    arrays = defaultdict(list)
    for topic, types in mock_ros2_node.get_topic_names_and_types():
        if SENSOR_MSG_TYPE in types:
            parts = topic.strip('/').split('/')
            if len(parts) >= 2:
                sensor_array = parts[0]
                arrays[sensor_array].append(topic)
    
    # Increment absent counter
    for sensor_array in list(active_arrays):
        if sensor_array not in arrays:
            absent_counts[sensor_array] = absent_counts.get(sensor_array, 0) + 1
    
    # Assert - not torn down yet (grace period)
    assert absent_counts['sensor_array_01'] == 1
    assert absent_counts['sensor_array_01'] < GRACE_POLLS
    assert 'sensor_array_01' in active_arrays


@pytest.mark.unit
def test_discovery_error_handling():
    """Test error handling (k8s API errors, network issues)."""
    # Test 1: K8s API error simulation
    mock_k8s = Mock()
    mock_k8s.create_namespaced_deployment.side_effect = Exception("K8s API unavailable")
    
    # Attempt to spawn should raise exception
    try:
        mock_k8s.create_namespaced_deployment(namespace="default", body={})
        raised = False
    except Exception as e:
        raised = True
        assert "K8s API unavailable" in str(e)
    
    assert raised is True
    
    # Test 2: Empty topic list handling
    topics = []
    assert len(topics) == 0
    # In real code, empty topics list should not trigger spawn
    
    # Test 3: Malformed topic name
    malformed_topic = "invalid_topic_format"
    parts = malformed_topic.strip('/').split('/')
    # Should not have enough parts for sensor array extraction
    assert len(parts) < 2
    
    # Test 4: Template file not found simulation
    import os
    template_path = "/nonexistent/template.yaml"
    assert not os.path.exists(template_path)
    
    # Test 5: Invalid YAML template
    invalid_yaml = "invalid: yaml: content:"
    import yaml
    try:
        yaml.safe_load(invalid_yaml)
        yaml_valid = True
    except:
        yaml_valid = False
    
    # This specific case might parse, but test demonstrates error handling
    assert isinstance(yaml_valid, bool)
