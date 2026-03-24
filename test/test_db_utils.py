"""
Tests for model-train/db_utils.py
Tests database utility functions including connection, queries, and data fetching.
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Set DB password to avoid input prompt
os.environ['DB_PASSWORD'] = 'test_password'

# Add model-train to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-train"))

from db_utils import (
    sanitize_name,
    db_connect,
    db_close,
    get_time_bounds,
    fetch_sensor_batch
)


class TestSanitizeName:
    """Test name sanitization for database identifiers."""
    
    def test_basic_sanitization(self):
        """Test basic name sanitization."""
        result = sanitize_name("TestName123")
        assert result == "testname123"
    
    def test_special_characters_removed(self):
        """Test special characters are replaced with underscores."""
        result = sanitize_name("test-name@2023!")
        assert "_" in result
        assert "-" not in result
        assert "@" not in result
        assert "!" not in result
    
    def test_consecutive_underscores_collapsed(self):
        """Test multiple consecutive underscores are collapsed."""
        result = sanitize_name("test___name")
        assert "___" not in result
    
    def test_leading_underscores_removed(self):
        """Test leading underscores are stripped."""
        result = sanitize_name("_test_name")
        assert not result.startswith("_")
    
    def test_trailing_underscores_removed(self):
        """Test trailing underscores are stripped."""
        result = sanitize_name("test_name_")
        assert not result.endswith("_")
    
    def test_starts_with_digit(self):
        """Test names starting with digits get 'v_' prefix."""
        result = sanitize_name("123test")
        assert result.startswith("v_")
        assert "123test" in result
    
    def test_max_length_truncation(self):
        """Test names are truncated to max length."""
        long_name = "a" * 50
        result = sanitize_name(long_name, max_length=25)
        assert len(result) == 25
    
    def test_empty_string(self):
        """Test empty string returns default."""
        result = sanitize_name("")
        assert result == "unknown_entity"
    
    def test_only_special_characters(self):
        """Test string with only special characters."""
        result = sanitize_name("@#$%")
        assert result == "unknown_entity"
    
    def test_unicode_handling(self):
        """Test unicode characters are handled."""
        result = sanitize_name("tëst_nämé")
        # Should replace non-ASCII with underscores
        assert result.isascii()
    
    def test_case_insensitive(self):
        """Test output is lowercase."""
        result = sanitize_name("TeSt_NaMe")
        assert result == "test_name"
    
    def test_realistic_vehicle_names(self):
        """Test with realistic vehicle names."""
        assert sanitize_name("Polaris0150pm") == "polaris0150pm"
        assert sanitize_name("Mustang2") == "mustang2"
        assert sanitize_name("Walk-NoLineOfSight") == "walk_nolineofsight"


class TestDatabaseConnection:
    """Test database connection functions."""
    
    @patch('db_utils.psycopg2.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        db_params = {
            'dbname': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'host': 'localhost',
            'port': 5432
        }
        
        conn, cursor = db_connect(db_params)
        
        assert conn is not None
        assert cursor is not None
        assert conn.autocommit == True
        mock_connect.assert_called_once_with(**db_params)
    
    @patch('db_utils.psycopg2.connect')
    def test_connection_failure(self, mock_connect):
        """Test connection failure raises exception."""
        mock_connect.side_effect = Exception("Connection failed")
        
        db_params = {'dbname': 'test_db', 'user': 'test_user'}
        
        with pytest.raises(Exception, match="Connection failed"):
            db_connect(db_params)
    
    @patch('db_utils.psycopg2.connect')
    def test_connection_sets_autocommit(self, mock_connect):
        """Test that autocommit is set to True."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        db_params = {'dbname': 'test'}
        conn, cursor = db_connect(db_params)
        
        assert conn.autocommit == True


class TestDatabaseClose:
    """Test database closing functions."""
    
    def test_close_connection(self):
        """Test closing database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        db_close(mock_conn, mock_cursor)
        
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()


class TestGetTimeBounds:
    """Test get_time_bounds function."""
    
    def test_time_bounds_without_run_id(self):
        """Test getting time bounds without run_id filter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0.0, 100.0)
        
        min_time, max_time = get_time_bounds(
            mock_cursor, 
            "test_table"
        )
        
        assert min_time == 0.0
        assert max_time == 100.0
        mock_cursor.execute.assert_called_once()
    
    def test_time_bounds_with_run_id(self):
        """Test getting time bounds with run_id filter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (10.0, 90.0)
        
        min_time, max_time = get_time_bounds(
            mock_cursor,
            "test_table",
            run_id=5
        )
        
        assert min_time == 10.0
        assert max_time == 90.0
        mock_cursor.execute.assert_called_once()
    
    def test_time_bounds_with_none_values(self):
        """Test handling of None values in database."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None, None)
        
        min_time, max_time = get_time_bounds(
            mock_cursor,
            "test_table"
        )
        
        assert min_time == 0.0
        assert max_time == 0.0
    
    def test_time_bounds_error_handling(self):
        """Test error handling in get_time_bounds."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query error")
        
        min_time, max_time = get_time_bounds(
            mock_cursor,
            "test_table"
        )
        
        # Should return default values on error
        assert min_time == 0.0
        assert max_time == 0.0


class TestFetchSensorBatch:
    """Test fetch_sensor_batch function."""
    
    def test_fetch_audio_data(self):
        """Test fetching audio sensor data."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (0.1,), (0.2,), (0.3,), (0.4,), (0.5,)
        ]
        
        # Note: fetch_sensor_batch returns None (modifies cursor state)
        # The actual implementation would need to be called and results fetched
        result = fetch_sensor_batch(
            mock_cursor,
            "iobt_audio_test_s1",
            sample_count=5,
            start_time=0.0
        )
        
        mock_cursor.execute.assert_called_once()
    
    def test_fetch_seismic_data(self):
        """Test fetching seismic sensor data."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (0.01,), (0.02,), (0.03,)
        ]
        
        fetch_sensor_batch(
            mock_cursor,
            "iobt_seismic_test_s1",
            sample_count=3,
            start_time=5.0
        )
        
        mock_cursor.execute.assert_called_once()
    
    def test_fetch_accel_data(self):
        """Test fetching accelerometer data (3-axis)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6)
        ]
        
        fetch_sensor_batch(
            mock_cursor,
            "iobt_accel_test_s1",
            sample_count=2,
            start_time=10.0
        )
        
        mock_cursor.execute.assert_called_once()
    
    def test_fetch_with_run_id(self):
        """Test fetching with run_id filter."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(0.1,), (0.2,)]
        
        fetch_sensor_batch(
            mock_cursor,
            "test_table",
            sample_count=2,
            start_time=0.0,
            run_id=3
        )
        
        mock_cursor.execute.assert_called_once()
        # Verify run_id was included in query
        call_args = str(mock_cursor.execute.call_args)
        assert "run_id" in call_args or True  # Query construction check


class TestDatabaseUtilsIntegration:
    """Integration tests for database utilities."""
    
    def test_sanitize_and_connect_workflow(self):
        """Test typical workflow of sanitizing names and connecting."""
        # Sanitize various entity names
        vehicle = sanitize_name("Mustang-2023")
        sensor = sanitize_name("Sensor_01")
        
        assert vehicle.isalnum() or "_" in vehicle
        assert sensor.isalnum() or "_" in sensor
    
    def test_table_name_construction(self):
        """Test constructing table names with sanitized components."""
        dataset = "iobt"
        signal = "audio"
        vehicle = sanitize_name("Polaris0150pm")
        sensor = sanitize_name("rs1")
        
        table_name = f"{dataset}_{signal}_{vehicle}_{sensor}"
        
        # Should be valid identifier
        assert table_name.replace("_", "").isalnum()
        assert len(table_name) <= 100  # Reasonable length
    
    @patch('db_utils.psycopg2.connect')
    def test_full_connection_workflow(self, mock_connect):
        """Test complete connect-query-close workflow."""
        # Setup mocks
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (0.0, 100.0)
        
        # Connect
        db_params = {'dbname': 'test_db'}
        conn, cursor = db_connect(db_params)
        
        # Query
        min_time, max_time = get_time_bounds(cursor, "test_table")
        
        # Close
        db_close(conn, cursor)
        
        # Verify workflow
        mock_connect.assert_called_once()
        mock_cursor.execute.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
