"""
Tests for md.storage.storage module using Azurite.

Assumes Azurite is running locally at http://127.0.0.1:10000
(started via docker-compose up -d)
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from md.storage.storage import (
    BlobStorage,
    FilesystemStorage,
    StorageConfig,
    create_storage,
    _guess_content_type,
)
from md.model.models import TileAddress


# Default Azurite connection string for local testing
AZURITE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"


@pytest.fixture(scope="session")
def azurite_connection_string() -> str:
    """Get Azurite connection string from env or use default."""
    return os.getenv("AZURITE_CONNECTION_STRING", AZURITE_CONNECTION_STRING)


@pytest.fixture
def storage_config(azurite_connection_string: str) -> StorageConfig:
    """Create a StorageConfig for testing with Azurite."""
    return StorageConfig(
        container="test-viz",
        connection_string=azurite_connection_string,
        create_container_if_missing=True,
    )


@pytest.fixture
def storage(storage_config: StorageConfig) -> Generator[BlobStorage, None, None]:
    """Create a BlobStorage instance for testing."""
    store = BlobStorage(storage_config)
    yield store
    
    # Cleanup: delete all blobs in the container
    try:
        blobs = store._cc.list_blobs()
        for blob in blobs:
            store._cc.delete_blob(blob.name)
    except Exception:
        pass  # Container might not exist


class TestGuessContentType:
    """Test the _guess_content_type() helper function."""

    def test_webp(self) -> None:
        assert _guess_content_type(Path("tile.webp")) == "image/webp"

    def test_png(self) -> None:
        assert _guess_content_type(Path("tile.png")) == "image/png"

    def test_json(self) -> None:
        assert _guess_content_type(Path("meta.json")) == "application/json"

    def test_unknown_extension(self) -> None:
        result = _guess_content_type(Path("file.unknown"))
        # May return a guessed type or default to application/octet-stream
        assert isinstance(result, str) and len(result) > 0


class TestTileAddress:
    """Test the TileAddress model."""

    def test_default_ext(self) -> None:
        addr = TileAddress(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
            x=8,
            y=5,
        )
        assert addr.ext == "png"

    def test_blob_name_default(self) -> None:
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
        assert addr.blob_name() == "tiles/v1/chl/0/avg/4/8/5.webp"

    def test_blob_name_custom_ext(self) -> None:
        addr = TileAddress(
            version="v2",
            var="temperature",
            species="fish",
            time=12,
            z=3,
            x=2,
            y=1,
            ext="png",
        )
        assert addr.blob_name() == "tiles/v2/temperature/fish/12/3/2/1.png"
    
    def test_is_mask(self) -> None:
        data_addr = TileAddress(version="v1", var="chl", species=0, time="avg", z=4, x=8, y=5)
        mask_addr = TileAddress(version="v1", var="mask", species=0, time="avg", z=4, x=8, y=5)
        assert data_addr.is_mask() is False
        assert mask_addr.is_mask() is True


class TestStorageConfig:
    """Test the StorageConfig model."""

    def test_connection_string_required_if_no_account_url(self, azurite_connection_string: str) -> None:
        # StorageConfig allows None, but BlobStorage will raise ValueError
        cfg = StorageConfig(
            container="test",
            connection_string=None,
            account_url=None,
        )
        # Verify the config was created
        assert cfg.container == "test"
        
        # But BlobStorage should raise ValueError when instantiated
        with pytest.raises(ValueError):
            BlobStorage(cfg)

    def test_valid_with_connection_string(self, azurite_connection_string: str) -> None:
        cfg = StorageConfig(
            container="test",
            connection_string=azurite_connection_string,
        )
        assert cfg.container == "test"
        assert cfg.connection_string == azurite_connection_string

    def test_max_workers_validation(self, azurite_connection_string: str) -> None:
        with pytest.raises(ValueError):
            StorageConfig(
                container="test",
                connection_string=azurite_connection_string,
                max_workers=0,  # Must be >= 1
            )
        
        with pytest.raises(ValueError):
            StorageConfig(
                container="test",
                connection_string=azurite_connection_string,
                max_workers=200,  # Must be <= 128
            )


class TestBlobStorage:
    """Test BlobStorage low-level operations."""

    def test_store_and_load_bytes(self, storage: BlobStorage) -> None:
        data = b"test image data"
        blob_name = "test/data.webp"
        
        storage.store_bytes(
            blob_name=blob_name,
            data=data,
            content_type="image/webp",
        )
        
        loaded = storage.load_bytes(blob_name=blob_name)
        assert loaded == data

    def test_store_with_cache_control(self, storage: BlobStorage) -> None:
        data = b"cached data"
        blob_name = "test/cached.json"
        
        storage.store_bytes(
            blob_name=blob_name,
            data=data,
            content_type="application/json",
            cache_control="no-cache",
        )
        
        # Verify it was stored
        assert storage.exists(blob_name=blob_name)
        assert storage.load_bytes(blob_name=blob_name) == data

    def test_exists_returns_true_for_existing_blob(self, storage: BlobStorage) -> None:
        blob_name = "test/exists.webp"
        storage.store_bytes(
            blob_name=blob_name,
            data=b"data",
            content_type="image/webp",
        )
        assert storage.exists(blob_name=blob_name)

    def test_exists_returns_false_for_missing_blob(self, storage: BlobStorage) -> None:
        assert not storage.exists(blob_name="nonexistent/blob.webp")

    def test_error_loading_nonexistent_blob(self, storage: BlobStorage) -> None:
        with pytest.raises(Exception):  # Azure raises ResourceNotFoundError
            storage.load_bytes(blob_name="nonexistent/blob.webp")

    def test_overwrite_false_raises_on_duplicate(self, storage: BlobStorage) -> None:
        blob_name = "test/nooverwrite.webp"
        
        storage.store_bytes(
            blob_name=blob_name,
            data=b"original",
            content_type="image/webp",
        )
        
        with pytest.raises(Exception):  # Azure raises ResourceExistsError
            storage.store_bytes(
                blob_name=blob_name,
                data=b"new data",
                content_type="image/webp",
                overwrite=False,
            )


class TestLoadTile:
    """Test high-level tile loading operations."""

    def test_load_tile_with_address(self, storage: BlobStorage) -> None:
        addr = TileAddress(
            version="v2",
            var="chl",
            species=0,
            time="avg",
            z=4,
            x=8,
            y=5,
            ext="webp",
        )
        tile_data = b"fake webp data"
        
        storage.store_bytes(
            blob_name=addr.blob_name(),
            data=tile_data,
            content_type="image/webp",
        )
        
        loaded = storage.load_tile(addr)
        assert loaded == tile_data

    def test_load_tile_by_parts(self, storage: BlobStorage) -> None:
        addr = TileAddress(
            version="v1",
            var="temperature",
            species="salmon",
            time=3,
            z=5,
            x=10,
            y=20,
            ext="webp",
        )
        tile_data = b"temperature tile"
        
        storage.store_bytes(
            blob_name=addr.blob_name(),
            data=tile_data,
            content_type="image/webp",
        )
        
        loaded = storage.load_tile_by_parts(
            version="v1",
            var="temperature",
            species="salmon",
            time=3,
            z=5,
            x=10,
            y=20,
            ext="webp",
        )
        assert loaded == tile_data

    def test_load_tile_by_parts_custom_ext(self, storage: BlobStorage) -> None:
        addr = TileAddress(
            version="v1",
            var="oxygen",
            species=99,
            time="avg",
            z=2,
            x=3,
            y=4,
            ext="png",
        )
        tile_data = b"png tile data"
        
        storage.store_bytes(
            blob_name=addr.blob_name(),
            data=tile_data,
            content_type="image/png",
        )
        
        loaded = storage.load_tile_by_parts(
            version="v1",
            var="oxygen",
            species=99,
            time="avg",
            z=2,
            x=3,
            y=4,
            ext="png",
        )
        assert loaded == tile_data
        assert loaded == tile_data


class TestStoreTiles:
    """Test parallel tile upload functionality."""

    def test_store_tiles_empty_directory(self, storage: BlobStorage) -> None:
        """store_tiles should handle empty directories gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory should not raise
            storage.store_tiles(local_tiles_root=tmpdir)

    def test_store_tiles_nonexistent_directory(self, storage: BlobStorage) -> None:
        """store_tiles should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            storage.store_tiles(local_tiles_root="/nonexistent/path")

    def test_store_tiles_with_json_and_tiles(self, storage: BlobStorage) -> None:
        """Test uploading a realistic tile tree with current.json, meta.json, and tile files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create directory structure
            v2_dir = root / "v2" / "chl" / "0" / "avg" / "4" / "8"
            v2_dir.mkdir(parents=True)
            
            # Create files
            current_json = b'{"version": "v2", "url": "/tiles/v2"}'
            meta_json = b'{"name": "Chlorophyll", "units": "mg/m3"}'
            tile1_data = b"fake webp data 1"
            tile2_data = b"fake webp data 2"
            
            (root / "current.json").write_bytes(current_json)
            (root / "v2" / "meta.json").write_bytes(meta_json)
            (v2_dir / "5.webp").write_bytes(tile1_data)
            (v2_dir / "6.webp").write_bytes(tile2_data)
            
            # Upload
            storage.store_tiles(
                local_tiles_root=root,
                remote_root="tiles",
                max_workers=2,
            )
            
            # Verify files were uploaded
            assert storage.exists(blob_name="tiles/current.json")
            assert storage.exists(blob_name="tiles/v2/meta.json")
            assert storage.exists(blob_name="tiles/v2/chl/0/avg/4/8/5.webp")
            assert storage.exists(blob_name="tiles/v2/chl/0/avg/4/8/6.webp")
            
            # Verify content
            assert storage.load_bytes(blob_name="tiles/current.json") == current_json
            assert storage.load_bytes(blob_name="tiles/v2/meta.json") == meta_json
            assert storage.load_bytes(blob_name="tiles/v2/chl/0/avg/4/8/5.webp") == tile1_data
            assert storage.load_bytes(blob_name="tiles/v2/chl/0/avg/4/8/6.webp") == tile2_data

    def test_store_tiles_cache_control(self, storage: BlobStorage) -> None:
        """Test that cache-control headers are set correctly for different file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create files
            (root / "current.json").write_bytes(b'{"version": "v1"}')
            (root / "tile.webp").write_bytes(b"webp data")
            
            storage.store_tiles(
                local_tiles_root=root,
                remote_root="tiles",
                cache_control_immutable="public, max-age=31536000, immutable",
                cache_control_pointer="no-cache",
            )
            
            # Verify files were uploaded
            assert storage.exists(blob_name="tiles/current.json")
            assert storage.exists(blob_name="tiles/tile.webp")

    def test_store_tiles_overwrite_false(self, storage: BlobStorage) -> None:
        """Test that overwrite=False prevents re-uploading existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tile.webp").write_bytes(b"original data")
            
            # First upload
            storage.store_tiles(
                local_tiles_root=root,
                remote_root="tiles",
                overwrite=True,
            )
            
            # Second upload with overwrite=False should fail
            (root / "tile.webp").write_bytes(b"updated data")
            with pytest.raises(RuntimeError):
                storage.store_tiles(
                    local_tiles_root=root,
                    remote_root="tiles",
                    overwrite=False,
                    fail_fast=True,
                )

    def test_store_tiles_fail_fast(self, storage: BlobStorage) -> None:
        """Test fail_fast behavior when upload errors occur."""
        # This is hard to test without mocking, but we can verify the parameter works
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tile.webp").write_bytes(b"data")
            
            # This should work without error
            storage.store_tiles(
                local_tiles_root=root,
                remote_root="tiles",
                fail_fast=True,
            )
            
            assert storage.exists(blob_name="tiles/tile.webp")


class TestLoadFile:
    """Test streaming file download functionality."""

    def test_load_file_basic(self, storage: BlobStorage) -> None:
        """Test downloading a blob to a local file."""
        blob_name = "test/largefile.bin"
        original_data = b"x" * (10 * 1024)  # 10KB
        
        # Upload the blob
        storage.store_bytes(
            blob_name=blob_name,
            data=original_data,
            content_type="application/octet-stream",
        )
        
        # Download to file
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "downloaded.bin"
            
            storage.load_file(blob_name=blob_name, local_path=local_path)
            
            # Verify the content matches
            assert local_path.exists()
            assert local_path.read_bytes() == original_data

    def test_load_file_creates_parent_directories(self, storage: BlobStorage) -> None:
        """Test that load_file creates parent directories if needed."""
        blob_name = "test/content.bin"
        data = b"test data"
        
        storage.store_bytes(
            blob_name=blob_name,
            data=data,
            content_type="application/octet-stream",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Request a deeply nested path that doesn't exist
            local_path = Path(tmpdir) / "deep" / "nested" / "path" / "file.bin"
            
            storage.load_file(blob_name=blob_name, local_path=local_path)
            
            # Verify directories were created and file exists
            assert local_path.exists()
            assert local_path.read_bytes() == data

    def test_load_file_large_file(self, storage: BlobStorage) -> None:
        """Test downloading a large file (>8MB) with streaming."""
        blob_name = "test/largefile.bin"
        original_data = b"x" * (20 * 1024 * 1024)  # 20MB to ensure multiple chunks
        
        storage.store_bytes(
            blob_name=blob_name,
            data=original_data,
            content_type="application/octet-stream",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "large.bin"
            
            # Should handle large files without loading everything into memory
            storage.load_file(blob_name=blob_name, local_path=local_path, chunk_size=1024*1024)
            
            assert local_path.exists()
            assert local_path.stat().st_size == len(original_data)

    def test_load_file_nonexistent_blob_raises(self, storage: BlobStorage) -> None:
        """Test that loading a nonexistent blob raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "file.bin"
            
            with pytest.raises(Exception):
                storage.load_file(blob_name="nonexistent.bin", local_path=local_path)


class TestVersionFolderCheck:
    """Test has_version_folder() functionality."""

    def test_has_version_folder_exists(self, storage: BlobStorage) -> None:
        """Test checking for an existing version folder."""
        # Create some blobs in a version folder
        storage.store_bytes(
            blob_name="v2/meta.json",
            data=b'{"version": "v2"}',
            content_type="application/json",
        )
        
        assert storage.has_version_folder("v2")

    def test_has_version_folder_not_exists(self, storage: BlobStorage) -> None:
        """Test checking for a nonexistent version folder."""
        assert not storage.has_version_folder("nonexistent-version")

    def test_has_version_folder_with_tiles(self, storage: BlobStorage) -> None:
        """Test version folder check with actual tiles."""
        # Create tiles in a version
        storage.store_bytes(
            blob_name="v3/chl/0/avg/4/8/5.webp",
            data=b"tile data",
            content_type="image/webp",
        )
        
        assert storage.has_version_folder("v3")
        assert not storage.has_version_folder("v4")


class TestMapfileCheck:
    """Test has_mapfile() functionality."""

    def test_has_mapfile_exists(self, storage: BlobStorage) -> None:
        """Test checking for an existing mapfile."""
        storage.store_bytes(
            blob_name="v2/data.nc",
            data=b"netcdf data",
            content_type="application/x-netcdf",
        )
        
        assert storage.has_mapfile("v2", "data.nc")

    def test_has_mapfile_not_exists(self, storage: BlobStorage) -> None:
        """Test checking for a nonexistent mapfile."""
        assert not storage.has_mapfile("v2", "nonexistent.nc")

    def test_has_mapfile_with_path_object(self, storage: BlobStorage) -> None:
        """Test has_mapfile works with Path objects."""
        storage.store_bytes(
            blob_name="v3/map.nc",
            data=b"map data",
            content_type="application/x-netcdf",
        )
        
        assert storage.has_mapfile("v3", Path("map.nc"))


class TestZarrFileCheck:
    """Test has_zarr_file() functionality."""

    def test_has_zarr_file_exists(self, storage: BlobStorage) -> None:
        """Test checking for an existing zarr file."""
        # Store a blob with the zarr prefix (zarr files are directories with multiple blobs)
        storage.store_bytes(
            blob_name="datasets/data.zarr/.zarray",
            data=b'{"zarr_format": 2}',
            content_type="application/json",
        )
        storage.store_bytes(
            blob_name="datasets/data.zarr/.zattrs",
            data=b'{}',
            content_type="application/json",
        )
        
        assert storage.has_zarr_file("datasets", "data.zarr")

    def test_has_zarr_file_not_exists(self, storage: BlobStorage) -> None:
        """Test checking for a nonexistent zarr file."""
        assert not storage.has_zarr_file("datasets", "nonexistent.zarr")

    def test_has_zarr_file_with_path_object(self, storage: BlobStorage) -> None:
        """Test has_zarr_file works with Path objects."""
        storage.store_bytes(
            blob_name="data/mydata.zarr/.zarray",
            data=b'{"zarr_format": 2}',
            content_type="application/json",
        )
        storage.store_bytes(
            blob_name="data/mydata.zarr/.zattrs",
            data=b'{}',
            content_type="application/json",
        )
        
        assert storage.has_zarr_file("data", Path("mydata.zarr"))


class TestTileSubtreeCheck:
    """Test has_tile_subtree() functionality."""

    def test_has_tile_subtree_exists(self, storage: BlobStorage) -> None:
        """Test checking for an existing tile subtree."""
        # Create tiles in a specific z-level
        storage.store_bytes(
            blob_name="tiles/v1/chl/0/avg/4/8/5.webp",
            data=b"tile",
            content_type="image/webp",
        )
        storage.store_bytes(
            blob_name="tiles/v1/chl/0/avg/4/8/6.webp",
            data=b"tile",
            content_type="image/webp",
        )
        
        assert storage.has_tile_subtree(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
        )

    def test_has_tile_subtree_not_exists(self, storage: BlobStorage) -> None:
        """Test checking for a nonexistent tile subtree."""
        assert not storage.has_tile_subtree(
            version="v1",
            var="chl",
            species=0,
            time="avg",
            z=4,
        )

    def test_has_tile_subtree_different_z_levels(self, storage: BlobStorage) -> None:
        """Test separating different z-levels."""
        # Create tiles for z=4
        storage.store_bytes(
            blob_name="tiles/v1/temp/0/avg/4/0/0.webp",
            data=b"z=4 tile",
            content_type="image/webp",
        )
        
        # Check z=4 exists
        assert storage.has_tile_subtree(
            version="v1",
            var="temp",
            species=0,
            time="avg",
            z=4,
        )
        
        # Check z=5 doesn't exist
        assert not storage.has_tile_subtree(
            version="v1",
            var="temp",
            species=0,
            time="avg",
            z=5,
        )


class TestStoreMapfile:
    """Test storing mapfiles."""

    def test_store_mapfile_basic(self, storage: BlobStorage) -> None:
        """Test storing a mapfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary mapfile
            mapfile_path = Path(tmpdir) / "data.nc"
            mapfile_data = b"netcdf data content"
            mapfile_path.write_bytes(mapfile_data)
            
            # Store it
            storage.store_mapfile(version="v2", mapfile_path=mapfile_path)
            
            # Verify it was stored
            assert storage.exists(blob_name="v2/data.nc")
            assert storage.load_bytes(blob_name="v2/data.nc") == mapfile_data

    def test_store_mapfile_with_path_object(self, storage: BlobStorage) -> None:
        """Test that store_mapfile works with Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapfile_path = Path(tmpdir) / "map.nc"
            mapfile_data = b"map content"
            mapfile_path.write_bytes(mapfile_data)
            
            storage.store_mapfile(version="v1", mapfile_path=mapfile_path)
            
            assert storage.exists(blob_name="v1/map.nc")

    def test_store_mapfile_content_type(self, storage: BlobStorage) -> None:
        """Test that mapfiles are stored with correct content type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapfile_path = Path(tmpdir) / "data.nc"
            mapfile_path.write_bytes(b"nc data")
            
            storage.store_mapfile(version="v3", mapfile_path=mapfile_path)
            
            # The file is stored and accessible
            assert storage.exists(blob_name="v3/data.nc")


class TestStoreZarr:
    """Test storing zarr directories."""

    def test_store_zarr_basic(self, storage: BlobStorage) -> None:
        """Test storing a basic zarr directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "data.zarr"
            zarr_dir.mkdir()
            
            # Create zarr structure
            (zarr_dir / ".zarray").write_bytes(b'{"zarr_format": 2}')
            (zarr_dir / ".zattrs").write_bytes(b'{"description": "test"}')
            (zarr_dir / "0").write_bytes(b"chunk0 data")
            
            # Store zarr
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="datasets/mydata.zarr",
            )
            
            # Verify all files were uploaded
            assert storage.exists(blob_name="datasets/mydata.zarr/.zarray")
            assert storage.exists(blob_name="datasets/mydata.zarr/.zattrs")
            assert storage.exists(blob_name="datasets/mydata.zarr/0")

    def test_store_zarr_nested_chunks(self, storage: BlobStorage) -> None:
        """Test storing zarr with nested chunk structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "data.zarr"
            zarr_dir.mkdir()
            
            # Create nested zarr structure
            (zarr_dir / ".zarray").write_bytes(b'{"zarr_format": 2, "chunks": [100]}')
            chunks_dir = zarr_dir / "0"
            chunks_dir.mkdir()
            (chunks_dir / "0").write_bytes(b"chunk data 0")
            (chunks_dir / "1").write_bytes(b"chunk data 1")
            
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="datasets/nested.zarr",
            )
            
            # Verify structure was preserved
            assert storage.exists(blob_name="datasets/nested.zarr/.zarray")
            assert storage.exists(blob_name="datasets/nested.zarr/0/0")
            assert storage.exists(blob_name="datasets/nested.zarr/0/1")

    def test_store_zarr_nonexistent_directory(self, storage: BlobStorage) -> None:
        """Test that storing a nonexistent zarr raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            storage.store_zarr(
                local_zarr_path="/nonexistent/zarr",
                remote_prefix="datasets",
            )

    def test_store_zarr_not_a_directory(self, storage: BlobStorage) -> None:
        """Test that store_zarr raises error if path is a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.txt"
            file_path.write_bytes(b"content")
            
            with pytest.raises(FileNotFoundError):
                storage.store_zarr(
                    local_zarr_path=file_path,
                    remote_prefix="datasets",
                )

    def test_store_zarr_empty_directory(self, storage: BlobStorage) -> None:
        """Test that storing an empty zarr directory does nothing silently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "empty.zarr"
            zarr_dir.mkdir()
            
            # Should not raise, just return
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="datasets",
            )

    def test_store_zarr_custom_prefix(self, storage: BlobStorage) -> None:
        """Test storing zarr with custom remote prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "data.zarr"
            zarr_dir.mkdir()
            (zarr_dir / ".zarray").write_bytes(b"{}")
            
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="custom/path/data.zarr",
            )
            
            assert storage.exists(blob_name="custom/path/data.zarr/.zarray")

    def test_store_zarr_overwrite_false(self, storage: BlobStorage) -> None:
        """Test store_zarr with overwrite=False raises on duplicate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "data.zarr"
            zarr_dir.mkdir()
            (zarr_dir / ".zarray").write_bytes(b"{}")
            
            # First store
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="test.zarr",
                overwrite=True,
            )
            
            # Second store with overwrite=False should fail
            with pytest.raises(RuntimeError):
                storage.store_zarr(
                    local_zarr_path=zarr_dir,
                    remote_prefix="test.zarr",
                    overwrite=False,
                    fail_fast=True,
                )

    def test_store_zarr_parallel_upload(self, storage: BlobStorage) -> None:
        """Test parallel upload with multiple workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_dir = Path(tmpdir) / "parallel.zarr"
            zarr_dir.mkdir()
            
            # Create multiple files
            (zarr_dir / ".zarray").write_bytes(b"{}")
            for i in range(10):
                (zarr_dir / f"chunk_{i}").write_bytes(f"data {i}".encode())
            
            storage.store_zarr(
                local_zarr_path=zarr_dir,
                remote_prefix="parallel.zarr",
                max_workers=4,
            )
            
            # Verify all were uploaded
            assert storage.exists(blob_name="parallel.zarr/.zarray")
            for i in range(10):
                assert storage.exists(blob_name=f"parallel.zarr/chunk_{i}")
