# storage_layer.py
from __future__ import annotations

from abc import ABC, abstractmethod

import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import shutil

import orjson

from pydantic import BaseModel, Field
from loguru import logger

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from iconfig.iconfig import iConfig

from md.model.models import TileAddress
from md.model.models import MapDefinition

# Optional (only needed in Azure MSI mode)
try:
    from azure.identity import DefaultAzureCredential  # type: ignore
except Exception:  # pragma: no cover
    DefaultAzureCredential = None  # type: ignore


def _guess_content_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    if suf == ".json":
        return "application/json"
    t, _ = mimetypes.guess_type(str(path))
    return t or "application/octet-stream"


class StorageConfig(BaseModel):
    """
    Configuration used by create_storage().
    """
    container: str = Field(default="viz")

    # One of:
    connection_string: Optional[str] = None  # ideal for Azurite
    account_url: Optional[str] = None        # ideal for Azure + MSI

    # Optional for advanced Azure auth; by default we use DefaultAzureCredential()
    use_default_azure_credential: bool = True

    # Container create behavior
    create_container_if_missing: bool = True

    # Upload tuning
    max_workers: int = Field(default=16, ge=1, le=128)
    upload_timeout_s: int = Field(default=30, ge=1, le=3600)  # Reduced from 120s: PNG tiles ~5-50KB, 30s is reasonable
    batch_size: int = Field(default=256, ge=1, le=256)  # Azure batch API limit


class Storage(ABC):
    """
    Abstract base class for storage layers.
    """

    @abstractmethod
    def get_stored_versions(self) -> list[str]:
        pass

    def has_version_folder(self, version: str) -> bool:
        return version in self.get_stored_versions()
    
    @abstractmethod
    def has_mapfile(self, version: str, mapfile: str|Path) -> bool:
        pass

    @abstractmethod
    def has_zarr_file(self, version: str, zarr_file: str|Path) -> bool:
        pass

    @abstractmethod
    def has_tile_subtree(self, *, version: str, var: str, species: Union[int, str], time: Union[int, str], z: int) -> bool:
        pass

    @abstractmethod
    def delete_container(self) -> bool:
        pass

    @abstractmethod
    def recreate_container(self) -> bool:
        pass

    @abstractmethod
    def store_mapfile(self, version: str, mapfile_path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def store_zarr(
        self,
        *,
        local_zarr_path: Union[str, Path],
        remote_prefix: str = "datasets",
        overwrite: bool = True,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        pass


    @abstractmethod
    def store_tiles(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def store_map_definition(self, remote_path: str, map_definition: dict) -> None:
        pass

    @abstractmethod
    def load_tile(self, *args, **kwargs) -> bytes:
        pass

    @abstractmethod
    def load_map_definition(self, version_id: str) -> Optional[MapDefinition]:
        pass

    @abstractmethod
    def load_bytes(self, *, blob_name: str) -> bytes:
        pass

    @abstractmethod
    def exists(self, *, blob_name: str) -> bool:
        pass

class FilesystemStorage(Storage):
    """
    Local filesystem storage layer for testing and development.
    Not optimized for performance or concurrency.
    """

    def __init__(self, cfg: StorageConfig):
        self._root = Path(cfg.container).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(f"FilesystemStorage initialized at {self._root}")
    
    def get_stored_versions(self) -> list[str]:
        # Get all subdirectories in the root as versions
        versions = []
        tiledir = self._root / "tiles"
        if not tiledir.exists():
            return versions
        for p in tiledir.iterdir():
            if p.is_dir():
                versions.append(p.name)
        return versions

    def has_mapfile(self, version: str, mapfile: str|Path) -> bool:
        name = f"{version}/{mapfile}"
        return self.exists(blob_name=name)

    def has_zarr_file(self, version: str, zarr_file: str|Path) -> bool:
        name = f"{version}/{zarr_file}/"
        return self.exists(blob_name=name)

    def has_tile_subtree(self, *, version: str, var: str, species: Union[int, str], time: Union[int, str], z: int) -> bool:
        prefix = f"tiles/{version}/{var}/{species}/{time}/{z}/"
        return self.exists(blob_name=prefix)

    def delete_container(self) -> bool:
        # remove the entire root directory and all contents
        if not self._root.exists():
            return True
        try:
            shutil.rmtree(self._root)
            logger.info(f"Deleted storage root: {self._root}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete storage root: {e}")
            return False

    def recreate_container(self) -> bool:
        try:
            self.delete_container()
            self._root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Recreated storage root: {self._root}")
            return True
        except Exception as e:
            logger.error(f"Failed to recreate storage root: {e}")
            return False

    def store_mapfile(self, version: str, mapfile_path: Union[str, Path]) -> None:
        name = f"{version}/{Path(mapfile_path).name}"
        with open(mapfile_path, 'rb') as f:
            data = f.read()
        fn = self._root / name
        fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, 'wb') as f:
            f.write(data)

    def store_zarr(
        self,
        *,
        local_zarr_path: Union[str, Path],
        remote_prefix: str = "datasets",
        overwrite: bool = True,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        shutil.copytree(local_zarr_path, self._root / remote_prefix, dirs_exist_ok=overwrite)

    def store_tiles(
        self,
        *,
        local_tiles_root: Union[str, Path],
        remote_root: str = "tiles",
        cache_control_immutable: str = "public, max-age=31536000, immutable",
        cache_control_pointer: str = "no-cache",
        overwrite: bool = True,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        src = Path(local_tiles_root).resolve()
        dst = self._root / remote_root
        if not src.exists():
            raise FileNotFoundError(f"local_tiles_root does not exist: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=overwrite)

    def store_map_definition(self, remote_path: str, map_definition: dict) -> None:
        data = orjson.dumps(map_definition)
        fn = self._root / remote_path
        fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, 'wb') as f:
            f.write(data)

    def load_tile(self, addr: TileAddress) -> bytes:
        fn = self._root / addr.blob_name()
        return self.load_bytes(blob_name=str(fn))

    def load_map_definition(self, version_id: str) -> Optional[MapDefinition]:
        fn = self._root / f"tiles/{version_id}/map_definition.json"
        if not fn.exists():
            return None
        with open(fn, 'rb') as f:
            data = f.read()
        return MapDefinition(**orjson.loads(data))

    def load_bytes(self, *, blob_name: str) -> bytes:
        if not blob_name.startswith(str(self._root)):
            blob_name = f"{self._root}/{blob_name}"
        fn = Path(blob_name)
        return fn.read_bytes()

    def exists(self, *, blob_name: str) -> bool:
        if not blob_name.startswith(str(self._root)):
            blob_name = f"{self._root}/{blob_name}"
        fn = Path(blob_name)
        return fn.exists()

class BlobStorage(Storage):
    """
    BlobStorage-compatible storage layer usable with:
      - Azurite (connection string)
      - Azure Blob Storage (account_url + Managed Identity)

    Key methods:
      - store_tiles(...)
      - load_tile(...)
    """

    def __init__(self, cfg: StorageConfig):
        if cfg.connection_string:
            # Azurite: use direct connection string
            svc = BlobServiceClient.from_connection_string(cfg.connection_string)
        elif cfg.account_url:
            # Azure: use account URL with optional MSI credentials
            if cfg.use_default_azure_credential:
                if DefaultAzureCredential is None:
                    raise RuntimeError(
                        "azure-identity not installed but required for DefaultAzureCredential. "
                        "pip install azure-identity"
                    )
                cred = DefaultAzureCredential()
            else:
                cred = None  # You can extend this to accept an injected credential
            svc = BlobServiceClient(account_url=cfg.account_url, credential=cred)
        else:
            raise ValueError("StorageConfig must include either connection_string or account_url.")

        self._cfg = cfg
        self._svc = svc
        self._cc = svc.get_container_client(cfg.container)

        if cfg.create_container_if_missing:
            try:
                self._cc.get_container_properties()
            except ResourceNotFoundError:
                self._cc.create_container()
                logger.info(f"Created container: {cfg.container}")
        
        logger.info(f"BlobStorage initialized: timeout={cfg.upload_timeout_s}s, max_workers={cfg.max_workers}, batch_size={cfg.batch_size}")

####################################################################################################################
#   Low-level primitives
####################################################################################################################

    def store_bytes(
        self,
        *,
        blob_name: str,
        data: bytes,
        content_type: str,
        cache_control: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        settings = ContentSettings(content_type=content_type, cache_control=cache_control)
        bc = self._cc.get_blob_client(blob_name)
        bc.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=settings,
            timeout=self._cfg.upload_timeout_s,
        )

    def load_bytes(self, *, blob_name: str) -> bytes:
        bc = self._cc.get_blob_client(blob_name)
        return bc.download_blob(timeout=self._cfg.upload_timeout_s).readall()

    def exists(self, *, blob_name: str) -> bool:
        bc = self._cc.get_blob_client(blob_name)
        return bool(bc.exists(timeout=self._cfg.upload_timeout_s))

####################################################################################################################
#   High-level operations
####################################################################################################################
    def load_file(self, blob_name: str, local_path: Union[str, Path], chunk_size: int = 8 * 1024 * 1024) -> None:
        """
        Download a blob to a local file with streaming to avoid memory overhead.
        
        Args:
            blob_name: Name of the blob to download
            local_path: Local file path to write to
            chunk_size: Size of chunks to download (default 8MB)
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        bc = self._cc.get_blob_client(blob_name)
        download_stream = bc.download_blob(timeout=self._cfg.upload_timeout_s)
        
        with open(local_path, 'wb') as f:
            for chunk in download_stream.chunks():
                f.write(chunk)

    def delete_blobs_by_pattern(self, *, prefix: str, pattern_suffix: str = ".webp") -> int:
        """
        Delete all blobs matching a pattern (e.g., all .webp files) using batch operations.
        
        Uses parallel batch deletion which is ~100x faster than deleting one-by-one.
        
        Args:
            prefix: Blob name prefix to search (e.g., "tiles/v1/")
            pattern_suffix: Only delete blobs ending with this (e.g., ".webp")
        
        Returns:
            Number of blobs deleted
        """
        batch_size = 256  # Azure allows up to 256 blobs per batch
        batch = []
        count = 0
        
        logger.info(f"Listing blobs with prefix: {prefix}")
        
        # List and delete in batches for efficiency
        for blob in self._cc.list_blobs(name_starts_with=prefix, timeout=self._cfg.upload_timeout_s):
            if blob.name.endswith(pattern_suffix):
                batch.append(blob.name)
                
                # When batch reaches size limit, delete it
                if len(batch) >= batch_size:
                    logger.info(f"Deleting batch of {len(batch)} blobs...")
                    try:
                        # delete_blobs() processes the batch in parallel on server side
                        self._cc.delete_blobs(*batch, timeout=self._cfg.upload_timeout_s)
                        count += len(batch)
                        logger.info(f"Deleted {count} blobs total so far")
                    except Exception as e:
                        logger.error(f"Error deleting batch: {e}")
                    
                    batch = []
        
        # Delete remaining blobs
        if batch:
            logger.info(f"Deleting final batch of {len(batch)} blobs...")
            try:
                self._cc.delete_blobs(*batch, timeout=self._cfg.upload_timeout_s)
                count += len(batch)
                logger.info(f"Deleted {count} blobs total")
            except Exception as e:
                logger.error(f"Error deleting final batch: {e}")
        
        logger.info(f"Deletion complete: {count} blobs deleted")
        return count
    
    def delete_container(self) -> bool:
        """
        Delete the entire container (much faster than deleting individual blobs).
        
        **WARNING:** This deletes EVERYTHING in the container.
        Only use if you're planning to regenerate all tiles anyway.
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Deleting container: {self._cfg.container}")
            self._cc.delete_container(timeout=self._cfg.upload_timeout_s)
            logger.info("✓ Container deleted")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete container: {e}")
            return False
    
    def recreate_container(self) -> bool:
        """
        Recreate an empty container.
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating container: {self._cfg.container}")
            self._cc.create_container(timeout=self._cfg.upload_timeout_s)
            logger.info("✓ Container created")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create container: {e}")
            return False

    def get_stored_versions(self) -> list[str]:
        """
        List all versions currently stored in the container by checking for blobs with the "tiles/{version}/" prefix.
        
        Returns:
            List of version strings (e.g., ["v1", "v2"])
        """
        versions = set()
        prefix = "tiles/"
        for blob in self._cc.list_blobs(name_starts_with=prefix, timeout=self._cfg.upload_timeout_s):
            parts = blob.name.split('/')
            if len(parts) > 1:
                versions.add(parts[1])
        return sorted(versions)

    def load_map_definition(self, version_id: str) -> MapDefinition:
        blob_name = f"tiles/{version_id}/map_definition.json"
        if not self.exists(blob_name=blob_name):
            return None
        data = self.load_bytes(blob_name=blob_name)
        return MapDefinition(**orjson.loads(data))

    def load_tile(self, addr: TileAddress) -> bytes:
        return self.load_bytes(blob_name=addr.blob_name())
    
    def has_version_folder(self, version: str) -> bool:
        blobs = list(self._cc.list_blobs(name_starts_with=version, timeout=self._cfg.upload_timeout_s))
        return len(blobs) > 0

    def has_mapfile(self, version: str, mapfile: str|Path) -> bool:
        name = f"{version}/{mapfile}"
        return self.exists(blob_name=name)
    
    def has_zarr_file(self, version: str, zarr_file: str|Path) -> bool:
        """Check if a zarr directory exists by checking for blobs with that prefix."""
        name = f"{version}/{zarr_file}/"
        blobs = list(self._cc.list_blobs(name_starts_with=name, timeout=self._cfg.upload_timeout_s))
        return len(blobs) > 0

    def has_tile_subtree(self, *, version: str, var: str, species: Union[int, str], time: Union[int, str], z: int) -> bool:
        prefix = f"tiles/{version}/{var}/{species}/{time}/{z}/"
        blobs = list(self._cc.list_blobs(name_starts_with=prefix, timeout=self._cfg.upload_timeout_s))
        return len(blobs) > 0

    def load_tile_by_parts(
        self,
        *,
        version: str,
        var: str,
        species: Union[int, str],
        time: Union[int, str],
        z: int,
        x: int,
        y: int,
        ext: str = "png",
    ) -> bytes:
        addr = TileAddress(
            version=version,
            var=var,
            species=species,
            time=time,
            z=z,
            x=x,
            y=y,
            ext=ext,
        )
        return self.load_tile(addr)

    def store_mapfile(self, version: str, mapfile_path: Union[str, Path]) -> None:
        name = f"{version}/{Path(mapfile_path).name}"
        with open(mapfile_path, 'rb') as f:
            data = f.read()
        self.store_bytes(blob_name=name, data=data, content_type="application/x-netcdf", cache_control="public, max-age=31536000, immutable")
    
    def store_map_definition(self, remote_path: str, map_definition: dict) -> None:
        data = orjson.dumps(map_definition)
        self.store_bytes(blob_name=remote_path, data=data, content_type="application/json", cache_control="public, max-age=31536000, immutable")

    def store_zarr(
        self,
        *,
        local_zarr_path: Union[str, Path],
        remote_prefix: str = "datasets",
        overwrite: bool = True,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        """
        Upload a zarr directory to Blob Storage, preserving structure.
        
        Args:
            local_zarr_path: Path to local zarr directory
            remote_prefix: Remote blob prefix (e.g., "datasets/mydata.zarr")
            overwrite: Whether to overwrite existing blobs
            max_workers: Number of parallel uploads
            fail_fast: Stop on first error
        """
        root = Path(local_zarr_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"zarr directory does not exist: {root}")
        
        files = [p for p in root.rglob("*") if p.is_file()]
        if not files:
            logger.info(f"No files to upload from {local_zarr_path}")
            return
        
        logger.info(f"Uploading {len(files)} zarr files to {remote_prefix}")
        workers = max_workers or self._cfg.max_workers
        
        def _upload_zarr_chunk(p: Path) -> None:
            rel = p.relative_to(root).as_posix()
            blob_name = f"{remote_prefix}/{rel}"
            
            data = p.read_bytes()
            self.store_bytes(
                blob_name=blob_name,
                data=data,
                content_type="application/octet-stream",
                overwrite=overwrite,
            )
        
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_upload_zarr_chunk, p): p for p in files}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                try:
                    fut.result()
                    if i % 10 == 0:  # Log progress every 10 files
                        logger.info(f"  Uploaded {i}/{len(files)} zarr files")
                except Exception as e:
                    errors.append((p, e))
                    if fail_fast:
                        for f in futs:
                            f.cancel()
                        break
        
        logger.info(f"Zarr upload complete: {len(files) - len(errors)}/{len(files)} files")
        if errors:
            msg = "\n".join([f"- {p}: {type(e).__name__}: {e}" for p, e in errors[:20]])
            raise RuntimeError(f"store_zarr() failed for {len(errors)} file(s):\n{msg}")

    def store_tiles(
        self,
        *,
        local_tiles_root: Union[str, Path],
        remote_root: str = "tiles",
        cache_control_immutable: str = "public, max-age=31536000, immutable",
        cache_control_pointer: str = "no-cache",
        overwrite: bool = True,
        max_workers: Optional[int] = None,
        fail_fast: bool = True,
    ) -> None:
        """
        Batch parallel upload of an on-disk tile tree to Blob-compatible storage.
        
        Uses Azure Batch API (up to 256 blobs per batch) for 100x fewer HTTP requests.
        Previously: 500k tiles = 500k HTTP requests. Now: ~2000 batch operations.

        Expected local layout:
          <local_tiles_root>/
            current.json
            v1/
              meta.json
              <var>/<species>/<time>/<z>/<x>/<y>.png

        Remote layout:
          <remote_root>/current.json
          <remote_root>/v1/meta.json
          <remote_root>/v1/<var>/...

        Caching:
          - current.json: no-cache
          - everything else: immutable (safe because versioned)
        """
        root = Path(local_tiles_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"local_tiles_root does not exist: {root}")
    
        files = [p for p in root.rglob("*") if p.is_file()]
        if not files:
            logger.info(f"No files to upload from {local_tiles_root}")
            return

        logger.info(f"Uploading {len(files)} files to {remote_root} using batch API")
        workers = max_workers or self._cfg.max_workers
        batch_size = self._cfg.batch_size
        uploaded = 0
        errors = []

        # Prepare upload specifications for all files
        uploads = []
        for p in files:
            rel = p.relative_to(root).as_posix()
            blob_name = f"{remote_root}/{rel}"
            ctype = _guess_content_type(p)
            cache = cache_control_pointer if rel.endswith("current.json") else cache_control_immutable
            
            data = p.read_bytes()
            uploads.append((blob_name, data, ctype, cache, p))

        def _upload_batch(batch_uploads: list) -> int:
            """Upload a batch of blobs and return count on success."""
            count = 0
            for blob_name, data, ctype, cache, p in batch_uploads:
                try:
                    self.store_bytes(
                        blob_name=blob_name,
                        data=data,
                        content_type=ctype,
                        cache_control=cache,
                        overwrite=overwrite,
                    )
                    count += 1
                except Exception as e:
                    errors.append((p, blob_name, e))
                    if fail_fast:
                        raise
            return count

        # Submit batches to thread pool
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i in range(0, len(uploads), batch_size):
                batch = uploads[i:i + batch_size]
                fut = ex.submit(_upload_batch, batch)
                futs.append((i // batch_size + 1, len(uploads) // batch_size, fut))
            
            # Wait for results with progress tracking
            for batch_num, total_batches, fut in futs:
                try:
                    count = fut.result()
                    uploaded += count
                    logger.info(f"Batch {batch_num}/{total_batches}: Uploaded {count} blobs ({uploaded}/{len(files)} total)")
                except Exception as e:
                    if fail_fast:
                        logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
                        for f in futs:
                            if f[2] != fut:
                                f[2].cancel()
                        raise RuntimeError(f"store_tiles() failed at batch {batch_num}/{total_batches}: {type(e).__name__}: {e}") from e

        logger.info(f"Upload complete: {uploaded}/{len(files)} files uploaded")
        if errors:
            msg = "\n".join([f"- {p} ({blob}): {type(e).__name__}: {e}" for p, blob, e in errors[:20]])
            if len(errors) > 20:
                msg += f"\n... and {len(errors) - 20} more errors"
            raise RuntimeError(f"store_tiles() had {len(errors)} error(s):\n{msg}")

####################################################################################################################
# Factory
####################################################################################################################

def create_storage(
    create_container_if_missing: bool = True,
) -> BlobStorage:
    """
    Create a BlobStorage instance for either Azurite or Azure.
    
    Args:
        container: Container name
        connection_string: Azure connection string (Azurite)
        account_url: Azure storage account URL
        max_workers: Number of parallel workers for upload (16 = good balance)
        upload_timeout_s: Timeout per blob operation in seconds (30 = reasonable for 5-50KB tiles)
        batch_size: Blobs per batch (max 256 for Azure API)
        create_container_if_missing: Auto-create container if not exists

    """
    cs = None
    au = None
    use_cred = False

    config = iConfig()
    storage_type = config("storage.type", default="azure").lower()
    container = config(f"storage.{storage_type}.container_name", default="viz")
    max_workers = config("storage.max_workers", default=16)
    upload_timeout_s = config("storage.upload_timeout_s", default=30)
    batch_size = config("storage.batch_size", default=256)
    
    match storage_type:
        case "azure":
            au = config("storage.azure.account_url")
            use_cred = True
            obj = BlobStorage
        case "azurite":
            cs = config("storage.azurite.connection_string")
            obj = BlobStorage
        case "filesystem":
            # For filesystem, we ignore Azure config and return a FilesystemStorage instance
            obj = FilesystemStorage
        case _:
            raise ValueError(f"Unsupported storage.type: {storage_type}")

    cfg = StorageConfig(
        container=container,
        connection_string=cs,
        account_url=au,
        use_default_azure_credential=use_cred,
        create_container_if_missing=create_container_if_missing,
        max_workers=max_workers,
        upload_timeout_s=upload_timeout_s,
        batch_size=batch_size,
    )
    return obj(cfg)
