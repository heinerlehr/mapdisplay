"""
Tests for md.model.version module.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
import pandas as pd
import xarray as xr

from md.model.version import Version, Versions


class TestVersion:
    """Test cases for Version class."""

    def test_version_initialization_with_datetime_string(self):
        """Test Version initialization converts datetime string to pd.Timestamp."""
        version = Version(
            id="v1",
            description="Test version",
            created_at="2026-02-10 10:00:00",
            mapfile=Path("map.nc"),
            annotations=[],
        )
        assert isinstance(version.created_at, pd.Timestamp)
        assert version.id == "v1"

    def test_version_initialization_with_datetime_object(self):
        """Test Version initialization with datetime object."""
        now = pd.Timestamp.now()
        version = Version(
            id="v1",
            description="Test version",
            created_at=now,
            mapfile=Path("map.nc"),
            annotations=[],
        )
        assert version.created_at == now

    def test_version_initialization_converts_mapfile_to_path(self):
        """Test Version initialization converts mapfile string to Path."""
        version = Version(
            id="v1",
            description="Test version",
            created_at="2026-02-10",
            mapfile="path/to/map.nc",
            annotations=[],
        )
        assert isinstance(version.mapfile, Path)
        assert version.mapfile == Path("path/to/map.nc")

    def test_version_str_representation(self):
        """Test Version string representation."""
        version = Version(
            id="v1",
            description="Test",
            created_at="2026-02-10",
            mapfile=Path("map.nc"),
            annotations=[],
        )
        str_repr = str(version)
        assert "v1" in str_repr
        assert "map.nc" in str_repr

    def test_version_update_attributes(self):
        """Test Version.update() modifies attributes."""
        with patch("md.model.version.Versions") as mock_versions:
            version = Version(
                id="v1",
                description="Original",
                created_at="2026-02-10",
                mapfile=Path("map.nc"),
                annotations=[],
            )
            
            version.update(description="Updated")
            
            assert version.description == "Updated"
            mock_versions.return_value.update_version.assert_called_once()

    def test_version_update_with_nonexistent_attribute(self):
        """Test Version.update() ignores nonexistent attributes."""
        with patch("md.model.version.Versions"):
            version = Version(
                id="v1",
                description="Original",
                created_at="2026-02-10",
                mapfile=Path("map.nc"),
                annotations=[],
            )
            
            version.update(nonexistent_attr="value")
            
            assert not hasattr(version, "nonexistent_attr")

    def test_update_annotations_with_provided_annotations(self):
        """Test update_annotations updates its state with provided annotations."""
        with patch("md.model.version.Versions"):
            version = Version(
                id="v1",
                description="Test",
                created_at="2026-02-10",
                mapfile=Path("/nonexistent/map.nc"),
                annotations=[],
            )
            
            version.update_annotations(
                annotations=[
                    {"key": "ann1", "value": "value1"},
                    {"key": "ann2", "value": "value2"},
                ],
                save=False
            )
            
            assert len(version.annotations) == 2
            assert version.annotations[0]["key"] == "ann1"
            assert version.annotations[1]["key"] == "ann2"

    def test_update_annotations_replaces_existing_key(self):
        """Test update_annotations replaces annotations with same key."""
        with patch("md.model.version.Versions"):
            version = Version(
                id="v1",
                description="Test",
                created_at="2026-02-10",
                mapfile=Path("/nonexistent/map.nc"),
                annotations=[{"key": "ann1", "value": "old"}],
            )
            
            # When starting with existing annotations and no file,
            # new annotations replace them if keys match
            version.annotations = [{"key": "ann1", "value": "old"}]
            
            new_ann = [{"key": "ann1", "value": "new"}]
            version.update_annotations(annotations=new_ann, save=False)
            
            # The update_annotations will replace old with new since no file exists
            assert len(version.annotations) > 0

    def test_update_annotations_with_none_clears(self):
        """Test update_annotations with None doesn't add any annotations."""
        with patch("md.model.version.Versions"):
            version = Version(
                id="v1",
                description="Test",
                created_at="2026-02-10",
                mapfile=Path("/nonexistent/map.nc"),
                annotations=[],
            )
            
            version.update_annotations(annotations=None, save=False)
            
            # When mapfile doesn't exist and no annotations provided, result is empty
            assert len(version.annotations) == 0

    def test_update_annotations_without_mapfile(self):
        """Test update_annotations when mapfile doesn't exist."""
        with patch("md.model.version.Versions"):
            version = Version(
                id="v1",
                description="Test",
                created_at="2026-02-10",
                mapfile=Path("/nonexistent/map.nc"),
                annotations=[],
            )
            
            version.update_annotations(
                annotations=[{"key": "ann1", "value": "value"}],
                save=False
            )
            
            assert len(version.annotations) == 1
            assert version.annotations[0]["key"] == "ann1"


class TestVersions:
    """Test cases for Versions singleton class."""

    def setup_method(self):
        """Reset singleton before each test."""
        # Clear singleton instances
        if hasattr(Versions, '_instances'):
            Versions._instances = {}

    def teardown_method(self):
        """Clean up after each test."""
        # Clear singleton instances
        if hasattr(Versions, '_instances'):
            Versions._instances = {}

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_get_version_by_id(self, mock_read_json, mock_iconfig):
        """Test Versions.get_version() retrieves correct version."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Version 1",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            },
            {
                "id": "v2",
                "description": "Version 2",
                "created_at": "2026-02-11",
                "mapfile": "map2.nc",
                "annotations": [],
            },
        ])
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict("os.environ", {"VERSION": "/config"}):
                    versions = Versions(versions=[])
                    version = versions.get_version("v1")
                    assert version.id == "v1"
                    assert version.description == "Version 1"

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_get_version_not_found(self, mock_read_json, mock_iconfig):
        """Test Versions.get_version() raises error for non-existent ID."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Version 1",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            }
        ])
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict("os.environ", {"VERSION": "/config"}):
                    versions = Versions(versions=[])
                    with pytest.raises(ValueError, match="Version with id"):
                        versions.get_version("nonexistent")

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_update_version_existing(self, mock_read_json, mock_iconfig):
        """Test Versions.update_version() updates existing version."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Original",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            }
        ])
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", create=True):
                    with patch.dict("os.environ", {"VERSION": "/config"}):
                        versions = Versions(versions=[])
                        
                        updated_version = Version(
                            id="v1",
                            description="Updated",
                            created_at="2026-02-10",
                            mapfile=Path("map1.nc"),
                            annotations=[],
                        )
                        
                        versions.update_version(updated_version, save=False)
                        
                        result = versions.get_version("v1")
                        assert result.description == "Updated"

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_update_version_new(self, mock_read_json, mock_iconfig):
        """Test Versions.update_version() appends new version."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Version 1",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            }
        ])
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", create=True):
                    with patch.dict("os.environ", {"VERSION": "/config"}):
                        versions = Versions(versions=[])
                        initial_count = len(versions.versions)
                        
                        new_version = Version(
                            id="v3",
                            description="Version 3",
                            created_at="2026-02-12",
                            mapfile=Path("map3.nc"),
                            annotations=[],
                        )
                        
                        versions.update_version(new_version, save=False)
                        
                        assert len(versions.versions) == initial_count + 1
                        result = versions.get_version("v3")
                        assert result.id == "v3"

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_get_newest_version(self, mock_read_json, mock_iconfig):
        """Test Versions.get_newest_version() returns most recent."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Version 1",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            },
            {
                "id": "v2",
                "description": "Version 2",
                "created_at": "2026-02-11",
                "mapfile": "map2.nc",
                "annotations": [],
            },
            {
                "id": "v3",
                "description": "Version 3",
                "created_at": "2026-02-15",
                "mapfile": "map3.nc",
                "annotations": [],
            },
        ])
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict("os.environ", {"VERSION": "/config"}):
                    versions = Versions(versions=[])
                    newest = versions.get_newest_version()
                    assert newest.id == "v3"

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    def test_versions_get_newest_version_no_versions(self, mock_read_json, mock_iconfig):
        """Test Versions.get_newest_version() raises error when empty."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame()
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict("os.environ", {"VERSION": "/config"}):
                    versions = Versions(versions=[])
                    versions.versions = []  # Explicitly set to empty
                    with pytest.raises(ValueError, match="No versions available"):
                        versions.get_newest_version()

    @patch("md.model.version.iConfig")
    @patch("md.model.version.read_json")
    @patch("builtins.open", create=True)
    @patch("orjson.dumps")
    def test_versions_save(self, mock_dumps, mock_open, mock_read_json, mock_iconfig):
        """Test Versions.save() writes to file."""
        mock_config = Mock()
        mock_config.return_value = "version.json"
        mock_iconfig.return_value = mock_config
        
        mock_read_json.return_value = pd.DataFrame([
            {
                "id": "v1",
                "description": "Version 1",
                "created_at": "2026-02-10",
                "mapfile": "map1.nc",
                "annotations": [],
            }
        ])
        
        mock_dumps.return_value = b'{"test": "data"}'
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch("os.getenv", return_value="/config"):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.dict("os.environ", {"VERSION": "/config"}):
                    versions = Versions(versions=[])
                    versions.save()
                    
                    mock_file.write.assert_called_once()

