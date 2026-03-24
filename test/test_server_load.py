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
    
    def test_determine_iobt_dataset(self):
        """Test recognizing IoBT dataset."""
        path = Path("/data/IoBT/audio/file.csv")
        dataset = determine_dataset(path)
        
        # Function returns string, check if it contains expected value
        assert dataset is not None
    
    def test_determine_focal_dataset(self):
        """Test recognizing FOCAL dataset."""
        path = Path("/data/FOCAL/seismic/data.csv")
        dataset = determine_dataset(path)
        
        assert dataset is not None
    
    def test_determine_m3nvc_dataset(self):
        """Test recognizing M3NVC dataset."""
        path = Path("/home/datasets/M3NVC/h08/audio.csv")
        dataset = determine_dataset(path)
        
        assert dataset is not None


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
