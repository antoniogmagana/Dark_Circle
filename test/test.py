"""
Main test runner for Dark_Circle test suite.
Validates imports and provides test metadata.
"""
import pytest
import sys
from pathlib import Path


def test_imports():
    """Test that all main modules can be imported."""
    try:
        # Test model-train imports
        sys.path.insert(0, str(Path(__file__).parent.parent / "model-train"))
        import config
        import models
        import dataset
        import data_generator
        import db_utils
        
        # Test ensemble-train imports
        sys.path.insert(0, str(Path(__file__).parent.parent / "ensemble-train"))
        import ensemble
        
        # Test server-load imports
        sys.path.insert(0, str(Path(__file__).parent.parent / "server-load"))
        import load_db
        
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_accessible():
    """Test that configuration is accessible."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "model-train"))
    from config import Config
    
    config = Config()
    assert config is not None
    assert hasattr(config, 'DEVICE')


def test_pytorch_available():
    """Test PyTorch is available."""
    try:
        import torch
        assert torch.__version__ is not None
    except ImportError:
        pytest.fail("PyTorch not available")


def test_numpy_available():
    """Test NumPy is available."""
    try:
        import numpy
        assert numpy.__version__ is not None
    except ImportError:
        pytest.fail("NumPy not available")


def test_pandas_available():
    """Test Pandas is available."""
    try:
        import pandas
        assert pandas.__version__ is not None
    except ImportError:
        pytest.fail("Pandas not available")


def test_workspace_structure():
    """Test workspace has expected structure."""
    workspace = Path(__file__).parent.parent
    
    # Check main directories exist
    assert (workspace / "model-train").exists()
    assert (workspace / "ensemble-train").exists()
    assert (workspace / "server-load").exists()
    assert (workspace / "test").exists()


def test_metadata():
    """Test suite metadata."""
    metadata = {
        "project": "Dark_Circle",
        "purpose": "Vehicle detection and classification",
        "test_framework": "pytest",
        "python_version": sys.version,
    }
    
    assert metadata["project"] == "Dark_Circle"
    assert metadata["test_framework"] == "pytest"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
