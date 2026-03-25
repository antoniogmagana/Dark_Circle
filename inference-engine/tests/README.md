# Inference-Engine Test Suite

Comprehensive test suite for the Dark_Circle inference-engine microservices pipeline.

## ✅ Current Status

**SensorBuffer Tests: 33/33 passing ✅**  
**Integration Tests: Pending ROS2/NATS/k8s mocking infrastructure**

## Overview

The inference-engine test suite covers:
- **SensorBuffer** (`test_buffer.py`) - ✅ **Complete** (33 comprehensive tests, all passing)
- **Discovery Node** (`test_discovery.py`) - 🚧 Stub (requires k8s mocking)
- **Ingestor Node** (`test_ingestor.py`) - 🚧 Stub (requires ROS2/NATS mocking)
- **Egress Node** (`test_egress.py`) - 🚧 Stub (requires NATS/ROS2 mocking)

## Quick Start

### Installation

```bash
cd /home/lvc_toolkit/project-files/Dark_Circle/inference-engine

# Install test dependencies
pip install -r tests/test_requirements.txt

# Install protobuf definitions
cd inference-protos
pip install -e .
cd ..
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (no external dependencies)
pytest tests/ -m unit -v

# Run SensorBuffer tests specifically
pytest tests/test_buffer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Generate HTML test report
pytest tests/ --html=test-report.html --self-contained-html
```

## Test Structure

```
inference-engine/
├── src/
│   ├── discovery/
│   ├── ingestor/
│   │   ├── main.py
│   │   └── buffer.py          # ← Tested (33 tests)
│   ├── infer_detect/
│   ├── infer_classify/
│   └── egress/
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures
    ├── pytest.ini              # Pytest configuration
    ├── test_requirements.txt   # Test dependencies
    ├── test_buffer.py          # ✅ Complete (33 tests)
    ├── test_discovery.py       # 🚧 Stubs
    ├── test_ingestor.py        # 🚧 Stubs
    └── test_egress.py          # 🚧 Stubs
```

## Test Coverage by Module

### ✅ SensorBuffer (`test_buffer.py`) - 33 tests

**Complete unit test coverage for sensor data buffering logic:**

#### Initialization (5 tests)
- Basic initialization
- Sample rate configuration
- Buffer size allocation
- ADC scale configuration
- Holding pen initialization

#### Basic Loading (4 tests)
- First data initialization
- Early data rejection
- Channel registration
- Data positioning by timestamp

#### Window Boundaries (4 tests)
- Acoustic triggers packaging
- Non-acoustic holding pen
- Holding pen transfer on reset
- Acoustic split across boundary

#### Multi-Channel Sync (3 tests)
- Multiple channels same window
- Different sample rates
- Three-axis accelerometer

#### ADC Normalization (3 tests)
- 16-bit acoustic normalization
- 24-bit seismic normalization  
- 24-bit accelerometer normalization

#### Protobuf Packaging (5 tests)
- Sensor ID in package
- Timestamp in package
- Channels list in package
- Tensor shapes validation
- Only active channels included

#### Edge Cases (6 tests)
- Exact window boundary
- Data far in future
- Single sample handling
- Empty array handling
- Reset clears channels
- Multiple windows sequence

#### Complex Scenarios (3 tests)
- Realistic acoustic chunks (100-sample chunks)
- Interleaved multi-channel
- Sparse seismic with dense acoustic

### 🚧 Discovery Node (`test_discovery.py`) - Stubs

**Pending k8s API mocking:**
- Topic discovery and parsing
- Kubernetes deployment spawning
- Deployment teardown with grace period
- Idempotent orchestration

### 🚧 Ingestor Node (`test_ingestor.py`) - Stubs

**Pending ROS2/NATS mocking:**
- Channel code mapping
- ROS2 subscription and callbacks
- Buffer integration
- NATS publishing

### 🚧 Egress Node (`test_egress.py`) - Stubs

**Pending NATS/ROS2 mocking:**
- Protobuf to ROS2 conversion
- NATS subscription
- Publishing logic (immediate on detection)
- Timestamp conversion

## Test Markers

Tests are categorized with markers for selective execution:

- `@pytest.mark.unit` - Pure logic tests (no external dependencies) ✅
- `@pytest.mark.integration` - Tests requiring mocked external systems 🚧
- `@pytest.mark.asyncio` - Async tests (NATS callbacks)
- `@pytest.mark.ros2` - Tests requiring ROS2 mocking
- `@pytest.mark.nats` - Tests requiring NATS mocking
- `@pytest.mark.k8s` - Tests requiring Kubernetes API mocking
- `@pytest.mark.slow` - Tests taking significant time

### Running by Marker

```bash
# Run only unit tests (ready to run)
pytest -m unit -v

# Skip integration tests
pytest -m "not integration" -v

# Run only async tests
pytest -m async -v
```

## Configuration

### pytest.ini

Configures test discovery, markers, and output options. Located at `tests/pytest.ini`.

### conftest.py

Provides shared fixtures:
- Sample data (acoustic, seismic, accel)
- Mock objects (ROS2 node, NATS client, k8s client)
- Protobuf helpers
- Assertions helpers

### test_requirements.txt

Core dependencies:
- pytest >= 7.4.0
- pytest-asyncio (for async NATS tests)
- pytest-mock
- numpy, protobuf
- pytest-html (for reports)

## Current Test Results

```bash
$ pytest tests/test_buffer.py -v

====================== test session starts ======================
collected 33 items

test_buffer.py::TestSensorBufferInit::test_init_basic PASSED
test_buffer.py::TestSensorBufferInit::test_init_sample_rates PASSED
test_buffer.py::TestSensorBufferInit::test_init_buffer_sizes PASSED
test_buffer.py::TestSensorBufferInit::test_init_adc_scales PASSED
test_buffer.py::TestSensorBufferInit::test_init_holding_pen PASSED
test_buffer.py::TestBasicLoading::test_load_first_data PASSED
test_buffer.py::TestBasicLoading::test_load_early_data_rejected PASSED
test_buffer.py::TestBasicLoading::test_load_channels_registered PASSED
test_buffer.py::TestBasicLoading::test_load_data_at_correct_position PASSED
... (24 more tests)

==================== 33 passed in 0.22s ====================
```

## Next Steps

### Phase 1: Complete Unit Tests ✅
- [x] SensorBuffer comprehensive tests (33 tests, all passing)

### Phase 2: Integration Test Infrastructure 🚧
- [ ] Set up ROS2 mocking (using unittest.mock or ros2-mock)
- [ ] Set up NATS mocking (using aioresponses or test NATS server)
- [ ] Set up Kubernetes API mocking (using kubernetes.client.mock)
- [ ] Implement Discovery node tests
- [ ] Implement Ingestor node tests
- [ ] Implement Egress node tests

### Phase 3: Model Inference Tests 📋
- [ ] Create tests for infer_detect (when implemented)
- [ ] Create tests for infer_classify (when implemented)
- [ ] Test model loading and inference
- [ ] Test preprocessing and postprocessing

### Phase 4: End-to-End Tests 📋
- [ ] Full pipeline test (ROS2 → Ingestor → NATS → Inference → Egress → ROS2)
- [ ] Performance tests (latency, throughput)
- [ ] Stress tests (high-frequency data streams)

## Writing New Tests

### Test Template

```python
import pytest
from component.module import ClassToTest

class TestFeature:
    """Test description."""
    
    @pytest.mark.unit
    def test_feature_name(self, fixture1, fixture2):
        \"\"\"Test specific behavior.\"\"\"
        # Arrange
        obj = ClassToTest()
        input_data = create_test_data()
        
        # Act
        result = obj.method(input_data)
        
        # Assert
        assert result == expected
```

### Using Fixtures

```python
def test_with_sensor_buffer(sensor_id, base_timestamp):
    \"\"\"Test using shared fixtures.\"\"\"
    buffer = SensorBuffer(sensor_id)
    buffer.load_buffer('acoustic', data, base_timestamp)
    assert buffer.start_time == base_timestamp
```

### Markers

```python
@pytest.mark.unit  # Pure logic, no mocking needed
def test_pure_function():
    pass

@pytest.mark.integration  # Requires mocking
@pytest.mark.ros2  # Specifically needs ROS2
def test_ros2_callback(mock_ros2_node):
    pass
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **AAA Pattern**: Arrange, Act, Assert
3. **Descriptive Names**: `test_acoustic_triggers_window_package`
4. **Docstrings**: Explain what behavior is being tested
5. **Fixtures**: Use shared fixtures from conftest.py
6. **Markers**: Tag tests appropriately for selective execution
7. **Coverage**: Aim for >90% coverage on pure logic

## Troubleshooting

### Import Errors

If you get import errors for protobuf definitions:

```bash
cd inference-protos
pip install -e .
```

### ROS2 Not Available

ROS2 tests are marked with `@pytest.mark.ros2` and skipped by default:

```bash
# Skip ROS2 tests
pytest -m "not ros2"
```

### Async Test Warnings

Install pytest-asyncio:

```bash
pip install pytest-asyncio
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Parent Project Tests](../test/) - Model-train test suite examples

## Contributing

When adding new features to inference-engine:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add new tests for new functionality
4. Update this README if adding new test modules
5. Maintain >80% code coverage

---

**Last Updated**: March 25, 2026  
**Status**: SensorBuffer tests complete (33/33), integration tests pending mocking infrastructure
