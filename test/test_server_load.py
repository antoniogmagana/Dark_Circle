"""
Tests for server-load/load_db.py
Tests database loading, dataset determination, and schema creation.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add server-load to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server-load"))

from load_db import (
    sanitize_name,
    determine_dataset,
    db_connect,
    db_close
)


class TestSanitizeName:
    """Test name sanitization for SQL Table names."""
    
    def test_sanitize_simple_name(self):
        """Test sanitizing simple valid name."""
        result = sanitize_name("valid_name")
        assert result == "valid_name"
    
    def test_sanitize_with_special_chars(self):
        """Test sanitizing name with special characters."""
        result = sanitize_name("table-name-with-dashes")
        assert "-" not in result
        assert result.replace("_", "").isalnum()
    
    def test_sanitize_with_spaces(self):
        """Test sanitizing name with spaces."""
        result = sanitize_name("table name with spaces")
        assert " " not in result
    
    def test_sanitize_uppercase(self):
        """Test sanitizing uppercase names."""
        result = sanitize_name("UPPERCASE_TABLE")
        # Should be lowercase
        assert result.islower()
    
    def test_sanitize_numbers(self):
        """Test sanitizing names with numbers."""
        result = sanitize_name("table_123")
        assert result == "table_123"
    
    def test_sanitize_leading_number(self):
        """Test sanitizing name starting with number."""
        result = sanitize_name("123_table")
        # Should handle leading number (adds v_ prefix)
        assert not result[0].isdigit() or result.startswith("v_")
    
    def test_sanitize_dots(self):
        """Test sanitizing names with dots (file extensions)."""
        result = sanitize_name("data.csv")
        assert "." not in result


class TestDetermineDataset:
    """Test dataset determination from file paths."""
    
    def test_determine_iobt_dataset(self, mocker):
        """Test recognizing IoBT dataset."""
        sensor_dir = Path("/data/IoBT/audio/sensor01")
        
        # Mock the exists() method for specific files
        def mock_exists(self):
            return str(self).endswith("aud16000.csv")
        
        mocker.patch.object(Path, 'exists', mock_exists)
        dataset = determine_dataset(sensor_dir)
        
        assert dataset == "iobt"
    
    def test_determine_focal_dataset(self, mocker):
        """Test recognizing FOCAL dataset."""
        sensor_dir = Path("/data/FOCAL/seismic/sensor02")
        
        # Mock the exists() method for specific files
        def mock_exists(self):
            return str(self).endswith("aud.csv")
        
        mocker.patch.object(Path, 'exists', mock_exists)
        dataset = determine_dataset(sensor_dir)
        
        assert dataset == "focal"
    
    def test_determine_m3nvc_dataset(self):
        """Test that M3NVC dataset returns None (not detected by file pattern)."""
        sensor_dir = Path("/home/datasets/M3NVC/h08/sensor")
        dataset = determine_dataset(sensor_dir)
        
        # M3NVC is not detected by determine_dataset - it only detects iobt/focal
        assert dataset is None


class TestDatabaseConnection:
    """Test database connection functions."""
    
    @patch('load_db.psycopg2.connect')
    def test_db_connect(self, mock_connect):
        """Test database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        conn, cursor = db_connect()
        
        assert conn is not None
        assert cursor is not None
        assert mock_connect.called
    
    def test_db_close(self):
        """Test database connection close."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        db_close(mock_conn, mock_cursor)
        
        assert mock_cursor.close.called
        assert mock_conn.close.called
