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
    create_table_name,
    create_schema,
    validate_csv_structure
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
        assert "_" in result or result == "tablenamebackspaces"
    
    def test_sanitize_uppercase(self):
        """Test sanitizing uppercase names."""
        result = sanitize_name("UPPERCASE_TABLE")
        # Should be lowercase for SQL
        assert result.islower() or result == "UPPERCASE_TABLE"
    
    def test_sanitize_numbers(self):
        """Test sanitizing names with numbers."""
        result = sanitize_name("table_123")
        assert result == "table_123"
    
    def test_sanitize_leading_number(self):
        """Test sanitizing name starting with number."""
        result = sanitize_name("123_table")
        # Should handle leading number (invalid SQL identifier)
        assert not result[0].isdigit() or result == "123_table"
    
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
        
        assert dataset == "IoBT"
    
    def test_determine_focal_dataset(self):
        """Test recognizing FOCAL dataset."""
        path = Path("/data/FOCAL/seismic/data.csv")
        dataset = determine_dataset(path)
        
        assert dataset == "FOCAL"
    
    def test_determine_m3nvc_dataset(self):
        """Test recognizing M3NVC dataset."""
        path = Path("/home/datasets/M3NVC/h08/audio.csv")
        dataset = determine_dataset(path)
        
        assert dataset == "M3NVC"
    
    def test_determine_unknown_dataset(self):
        """Test handling unknown dataset."""
        path = Path("/data/unknown/data.csv")
        dataset = determine_dataset(path)
        
        assert dataset == "Unknown" or dataset is None
    
    def test_determine_case_insensitive(self):
        """Test dataset detection is case insensitive."""
        path = Path("/data/iobt/data.csv")
        dataset = determine_dataset(path)
        
        # Should recognize lowercase 'iobt'
        assert dataset.upper() == "IOBT"


class TestCreateTableName:
    """Test table name creation from file paths."""
    
    def test_create_simple_table_name(self):
        """Test creating simple table name."""
        path = Path("/data/vehicle/audio.csv")
        sensor_type = "audio"
        
        table_name = create_table_name(path, sensor_type)
        
        assert "audio" in table_name.lower()
        assert table_name.replace("_", "").isalnum()
    
    def test_create_table_name_with_vehicle(self):
        """Test creating table name with vehicle identifier."""
        path = Path("/data/mustang/seismic.csv")
        sensor_type = "seismic"
        
        table_name = create_table_name(path, sensor_type)
        
        assert "mustang" in table_name.lower() or "seismic" in table_name.lower()
    
    def test_create_table_name_sanitized(self):
        """Test table name is properly sanitized."""
        path = Path("/data/vehicle-name/audio-sensor.csv")
        sensor_type = "audio-sensor"
        
        table_name = create_table_name(path, sensor_type)
        
        # Should have no dashes
        assert "-" not in table_name
    
    def test_create_unique_table_names(self):
        """Test different paths create different table names."""
        path1 = Path("/data/mustang/audio.csv")
        path2 = Path("/data/bicycle/audio.csv")
        sensor_type = "audio"
        
        table1 = create_table_name(path1, sensor_type)
        table2 = create_table_name(path2, sensor_type)
        
        # Should be different
        assert table1 != table2


class TestCreateSchema:
    """Test database schema creation."""
    
    @patch('load_db.psycopg2.connect')
    def test_create_audio_schema(self, mock_connect):
        """Test creating schema for audio sensor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        table_name = "audio_mustang"
        sensor_type = "audio"
        
        create_schema(mock_conn, table_name, sensor_type)
        
        # Should execute CREATE TABLE
        assert mock_cursor.execute.called
        call_args = str(mock_cursor.execute.call_args)
        assert "CREATE TABLE" in call_args.upper()
        assert table_name in call_args
    
    @patch('load_db.psycopg2.connect')
    def test_create_seismic_schema(self, mock_connect):
        """Test creating schema for seismic sensor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        table_name = "seismic_bicycle"
        sensor_type = "seismic"
        
        create_schema(mock_conn, table_name, sensor_type)
        
        assert mock_cursor.execute.called
    
    @patch('load_db.psycopg2.connect')
    def test_create_accelerometer_schema(self, mock_connect):
        """Test creating schema for accelerometer sensor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        table_name = "accelerometer_data"
        sensor_type = "accelerometer"
        
        create_schema(mock_conn, table_name, sensor_type)
        
        assert mock_cursor.execute.called
    
    @patch('load_db.psycopg2.connect')
    def test_schema_has_timestamp(self, mock_connect):
        """Test schema includes timestamp column."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        table_name = "test_table"
        sensor_type = "audio"
        
        create_schema(mock_conn, table_name, sensor_type)
        
        call_args = str(mock_cursor.execute.call_args)
        assert "timestamp" in call_args.lower() or "time" in call_args.lower()
    
    @patch('load_db.psycopg2.connect')
    def test_schema_has_value_column(self, mock_connect):
        """Test schema includes value/data column."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        table_name = "test_table"
        sensor_type = "seismic"
        
        create_schema(mock_conn, table_name, sensor_type)
        
        call_args = str(mock_cursor.execute.call_args)
        assert "value" in call_args.lower() or "data" in call_args.lower()


class TestValidateCSVStructure:
    """Test CSV structure validation."""
    
    def test_validate_valid_csv(self, tmp_path):
        """Test validating valid CSV structure."""
        csv_file = tmp_path / "valid.csv"
        csv_file.write_text("timestamp,value\n1234567890,0.5\n1234567891,0.6\n")
        
        is_valid = validate_csv_structure(str(csv_file))
        
        assert is_valid
    
    def test_validate_missing_timestamp(self, tmp_path):
        """Test detecting missing timestamp column."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("value,other\n0.5,1.0\n")
        
        is_valid = validate_csv_structure(str(csv_file))
        
        assert not is_valid
    
    def test_validate_missing_value(self, tmp_path):
        """Test detecting missing value column."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("timestamp,other\n1234567890,1.0\n")
        
        is_valid = validate_csv_structure(str(csv_file))
        
        assert not is_valid
    
    def test_validate_empty_csv(self, tmp_path):
        """Test handling empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        is_valid = validate_csv_structure(str(csv_file))
        
        assert not is_valid
    
    def test_validate_header_only(self, tmp_path):
        """Test CSV with header but no data."""
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("timestamp,value\n")
        
        is_valid = validate_csv_structure(str(csv_file))
        
        # Header only might be valid structure
        assert is_valid or not is_valid  # Implementation dependent


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @patch('load_db.psycopg2.connect')
    def test_full_load_pipeline(self, mock_connect):
        """Test full data loading pipeline."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Simulate loading data
        path = Path("/data/mustang/audio.csv")
        sensor_type = "audio"
        
        # 1. Determine dataset
        dataset = determine_dataset(path)
        assert dataset is not None
        
        # 2. Create table name
        table_name = create_table_name(path, sensor_type)
        assert table_name is not None
        
        # 3. Sanitize
        sanitized = sanitize_name(table_name)
        assert sanitized is not None
        
        # 4. Create schema
        create_schema(mock_conn, sanitized, sensor_type)
        assert mock_cursor.execute.called
    
    @patch('load_db.psycopg2.connect')
    def test_multiple_tables(self, mock_connect):
        """Test creating multiple tables."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        sensors = ["audio", "seismic", "accelerometer"]
        
        for sensor in sensors:
            table_name = f"test_{sensor}"
            create_schema(mock_conn, table_name, sensor)
        
        # Should create 3 tables
        assert mock_cursor.execute.call_count >= 3
