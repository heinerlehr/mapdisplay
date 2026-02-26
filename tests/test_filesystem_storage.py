"""
Tests for FilesystemStorage implementation.

Tests cover local filesystem storage layer for testing and development.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from md.storage.storage import (
    FilesystemStorage,
    StorageConfig,
)
from md.model.models import TileAddress


@pytest.fixture
def filesystem_storage_config() -> StorageConfig:
    """Create a StorageConfig for testing with filesystem storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = StorageConfig(
            container=tmpdir,
            create_container_if_missing=True,
        )
        yield cfg


@pytest.fixture
def filesystem_storage(filesystem_storage_config: StorageConfig) -> Generator[FilesystemStorage, None, None]:
    """Create a FilesystemStorage instance for testing."""
    store = FilesystemStorage(filesystem_storage_config)
    yield store
    
    # Cleanup handled by tempfile context manager


class TestFilesystemStorageBasics:
    """Test basic FilesystemStorage operations."""

    def test_init_creates_directory(self) -> None:
        """Test that FilesystemStorage creates container directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            container_path = Path(tmpdir) / "test_container"
            cfg = StorageConfig(container=str(container_path))
            
            storage = FilesystemStorage(cfg)
            
            assert container_path.exists()
            assert container_path.is_dir()

    def test_store_and_load_bytes(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing and loading bytes."""
        blob_name = "test/data.bin"
        test_data = b"test binary data"
        
        # Since FilesystemStorage doesn't have store_bytes, we test through exists/load_bytes pattern
        # by directly accessing the filesystem
        root = filesystem_storage._root
        file_path = root / blob_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(test_data)
        
        # Test exists
        assert filesystem_storage.exists(blob_name=blob_name)
        
        # Test load_bytes
        loaded = filesystem_storage.load_bytes(blob_name=blob_name)
        assert loaded == test_data

    def test_exists_returns_false_for_missing_file(self, filesystem_storage: FilesystemStorage) -> None:
        """Test that exists returns False for missing files."""
        assert not filesystem_storage.exists(blob_name="nonexistent/file.txt")

    def test_delete_container(self, filesystem_storage: FilesystemStorage) -> None:
        """Test deleting the container."""
        root = filesystem_storage._root
        assert root.exists()
        
        result = filesystem_storage.delete_container()
        
        assert result is True
        assert not root.exists()

    def test_recreate_container(self, filesystem_storage: FilesystemStorage) -> None:
        """Test recreating the container."""
        root = filesystem_storage._root
        original_path = root
        
        # Delete first
        filesystem_storage.delete_container()
        assert not root.exists()
        
        # Recreate
        result = filesystem_storage.recreate_container()
        
        assert result is True
        # Root might be different Path object but should point to same directory
        assert filesystem_storage._root.exists()


class TestFilesystemStorageVersions:
    """Test version management in FilesystemStorage."""

    def test_get_stored_versions_empty(self, filesystem_storage: FilesystemStorage) -> None:
        """Test getting stored versions when none exist."""
        versions = filesystem_storage.get_stored_versions()
        assert versions == []

    def test_get_stored_versions_with_tiles(self, filesystem_storage: FilesystemStorage) -> None:
        """Test getting stored versions with tile files."""
        root = filesystem_storage._root
        
        # Create version directories with tiles
        v1_dir = root / "tiles" / "v1" / "chl" / "0" / "avg" / "4" / "8"
        v1_dir.mkdir(parents=True, exist_ok=True)
        (v1_dir / "5.webp").write_bytes(b"tile data")
        
        v2_dir = root / "tiles" / "v2" / "temp" / "1" / "mean" / "3" / "10"
        v2_dir.mkdir(parents=True, exist_ok=True)
        (v2_dir / "12.webp").write_bytes(b"tile data")
        
        versions = filesystem_storage.get_stored_versions()
        assert sorted(versions) == ["v1", "v2"]

    def test_has_version_folder(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for version folders."""
        root = filesystem_storage._root
        v1_dir = root / "tiles" / "v1"
        v1_dir.mkdir(parents=True, exist_ok=True)
        (v1_dir / "meta.json").write_bytes(b"{}")
        
        assert filesystem_storage.has_version_folder("v1")
        assert not filesystem_storage.has_version_folder("v2")


class TestFilesystemStorageMapfiles:
    """Test mapfile operations in FilesystemStorage."""

    def test_has_mapfile_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for existing mapfile."""
        root = filesystem_storage._root
        mapfile_path = root / "v1" / "data.nc"
        mapfile_path.parent.mkdir(parents=True, exist_ok=True)
        mapfile_path.write_bytes(b"netcdf data")
        
        assert filesystem_storage.has_mapfile("v1", "data.nc")

    def test_has_mapfile_not_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for nonexistent mapfile."""
        assert not filesystem_storage.has_mapfile("v1", "nonexistent.nc")

    def test_store_mapfile(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing a mapfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary mapfile
            mapfile_path = Path(tmpdir) / "data.nc"
            mapfile_data = b"netcdf data content"
            mapfile_path.write_bytes(mapfile_data)
            
            # Store it
            filesystem_storage.store_mapfile(version="v1", mapfile_path=mapfile_path)
            
            # Verify it was stored
            assert filesystem_storage.has_mapfile("v1", "data.nc")
            stored_data = (filesystem_storage._root / "v1" / "data.nc").read_bytes()
            assert stored_data == mapfile_data

    def test_store_mapfile_with_path_object(self, filesystem_storage: FilesystemStorage) -> None:
        """Test that store_mapfile works with Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapfile_path = Path(tmpdir) / "map.nc"
            mapfile_data = b"map content"
            mapfile_path.write_bytes(mapfile_data)
            
            filesystem_storage.store_mapfile(version="v2", mapfile_path=Path(mapfile_path))
            
            assert filesystem_storage.has_mapfile("v2", "map.nc")


class TestFilesystemStorageZarr:
    """Test zarr operations in FilesystemStorage."""

    def test_has_zarr_file_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for existing zarr file."""
        root = filesystem_storage._root
        zarr_path = root / "datasets" / "data.zarr" / ".zarray"
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        zarr_path.write_bytes(b'{"zarr_format": 2}')
        
        assert filesystem_storage.has_zarr_file("datasets", "data.zarr/.zarray")

    def test_has_zarr_file_not_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for nonexistent zarr file."""
        assert not filesystem_storage.has_zarr_file("datasets", "nonexistent.zarr")

    def test_store_zarr(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing a zarr directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary zarr directory
            zarr_dir = Path(tmpdir) / "test.zarr"
            zarr_dir.mkdir()
            (zarr_dir / ".zarray").write_bytes(b'{"zarr_format": 2}')
            (zarr_dir / ".zattrs").write_bytes(b'{}')
            (zarr_dir / "data").write_bytes(b"chunk data")
            
            # Store it
            filesystem_storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="datasets/data.zarr",
            )
            
            # Verify all files were copied
            root = filesystem_storage._root
            assert (root / "datasets" / "data.zarr" / ".zarray").exists()
            assert (root / "datasets" / "data.zarr" / ".zattrs").exists()
            assert (root / "datasets" / "data.zarr" / "data").exists()

    def test_store_zarr_nonexistent_directory(self, filesystem_storage: FilesystemStorage) -> None:
        """Test that storing nonexistent zarr raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            filesystem_storage.store_zarr(
                local_zarr_path="/nonexistent/path",
                remote_prefix="datasets",
            )

    def test_store_zarr_nested_structure(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing zarr with nested chunk structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "nested.zarr"
            zarr_dir.mkdir()
            
            (zarr_dir / ".zarray").write_bytes(b'{}')
            chunks_dir = zarr_dir / "chunks"
            chunks_dir.mkdir()
            (chunks_dir / "0").write_bytes(b"chunk 0")
            (chunks_dir / "1").write_bytes(b"chunk 1")
            
            filesystem_storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="datasets/nested.zarr",
            )
            
            root = filesystem_storage._root
            assert (root / "datasets" / "nested.zarr" / ".zarray").exists()
            assert (root / "datasets" / "nested.zarr" / "chunks" / "0").exists()
            assert (root / "datasets" / "nested.zarr" / "chunks" / "1").exists()


class TestFilesystemStorageTiles:
    """Test tile operations in FilesystemStorage."""

    def test_has_tile_subtree_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for existing tile subtree."""
        root = filesystem_storage._root
        tile_dir = root / "tiles" / "v1" / "chl" / "0" / "avg" / "4" / "8"
        tile_dir.mkdir(parents=True, exist_ok=True)
        (tile_dir / "5.webp").write_bytes(b"tile data")
        
        assert filesystem_storage.has_tile_subtree(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
        )

    def test_has_tile_subtree_not_exists(self, filesystem_storage: FilesystemStorage) -> None:
        """Test checking for nonexistent tile subtree."""
        assert not filesystem_storage.has_tile_subtree(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
        )

    def test_load_tile(self, filesystem_storage: FilesystemStorage) -> None:
        """Test loading a tile."""
        root = filesystem_storage._root
        tile_dir = root / "tiles" / "v1" / "chl" / "0" / "avg" / "4" / "8"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_data = b"fake webp data"
        (tile_dir / "5.webp").write_bytes(tile_data)
        
        addr = TileAddress(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
            x=8,
            y=5,
            ext="webp",
        )
        
        loaded = filesystem_storage.load_tile(addr)
        assert loaded == tile_data

    def test_store_tiles(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing tiles from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary tile structure
            root = Path(tmpdir)
            v1_dir = root / "v1" / "chl" / "0" / "avg" / "4" / "8"
            v1_dir.mkdir(parents=True, exist_ok=True)
            
            tile1_data = b"tile 1"
            tile2_data = b"tile 2"
            (v1_dir / "5.webp").write_bytes(tile1_data)
            (v1_dir / "6.webp").write_bytes(tile2_data)
            
            (root / "current.json").write_bytes(b'{"version": "v1"}')
            
            # Store tiles
            filesystem_storage.store_tiles(
                local_tiles_root=root,
                remote_root="tiles",
            )
            
            # Verify tiles were copied
            storage_root = filesystem_storage._root
            assert (storage_root / "tiles" / "v1" / "chl" / "0" / "avg" / "4" / "8" / "5.webp").exists()
            assert (storage_root / "tiles" / "v1" / "chl" / "0" / "avg" / "4" / "8" / "6.webp").exists()
            assert (storage_root / "tiles" / "current.json").exists()

    def test_store_tiles_nonexistent_directory(self, filesystem_storage: FilesystemStorage) -> None:
        """Test that storing tiles from nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            filesystem_storage.store_tiles(
                local_tiles_root="/nonexistent/path",
                remote_root="tiles",
            )


class TestFilesystemStorageMapDefinition:
    """Test map definition operations in FilesystemStorage."""

    def test_store_and_load_map_definition(self, filesystem_storage: FilesystemStorage) -> None:
        """Test storing and loading map definition."""
        import orjson
        from md.model.models import MapDefinition, VariableConfig
        
        # Create a map definition
        map_def = MapDefinition(
            version_id="v1",
            created_at="2024-01-01T00:00:00Z",
            mapfile="data.nc",
            variables=[
                VariableConfig(name="chl", colormap="viridis"),
                VariableConfig(name="temp", colormap="coolwarm"),
            ],
            mask=None,
            species=["salmon", "trout"],
            time_range=None,
            time_labels=None,
            zoom_levels=[0, 1, 2, 3, 4, 5],
        )
        
        # Store it
        filesystem_storage.store_map_definition(
            remote_path="tiles/v1/map_definition.json",
            map_definition=map_def.model_dump(),
        )
        
        # Verify it was stored
        root = filesystem_storage._root
        def_file = root / "tiles" / "v1" / "map_definition.json"
        assert def_file.exists()
        
        # Load and verify
        loaded_def = filesystem_storage.load_map_definition("v1")
        assert loaded_def is not None
        assert loaded_def.version_id == "v1"
        assert len(loaded_def.variables) == 2

    def test_load_nonexistent_map_definition(self, filesystem_storage: FilesystemStorage) -> None:
        """Test loading nonexistent map definition returns None."""
        loaded_def = filesystem_storage.load_map_definition("nonexistent")
        assert loaded_def is None


class TestFilesystemStorageCreationPatterns:
    """Test different ways to create and use FilesystemStorage."""

    def test_storage_with_explicit_path(self) -> None:
        """Test creating storage with explicit container path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            container_path = Path(tmpdir) / "my_storage"
            cfg = StorageConfig(container=str(container_path))
            
            storage = FilesystemStorage(cfg)
            
            assert storage._root == container_path.resolve()
            assert container_path.exists()

    def test_storage_with_relative_path(self) -> None:
        """Test creating storage with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = StorageConfig(container="./storage")
                
                storage = FilesystemStorage(cfg)
                assert storage._root.exists()
            finally:
                os.chdir(original_cwd)

    def test_multiple_storages_independent(self) -> None:
        """Test that multiple FilesystemStorage instances are independent."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            cfg1 = StorageConfig(container=tmpdir1)
            cfg2 = StorageConfig(container=tmpdir2)
            
            storage1 = FilesystemStorage(cfg1)
            storage2 = FilesystemStorage(cfg2)
            
            # Write to storage1
            root1 = storage1._root / "test.txt"
            root1.write_bytes(b"storage1 data")
            
            # Verify storage2 doesn't have it
            root2 = storage2._root / "test.txt"
            assert not root2.exists()
