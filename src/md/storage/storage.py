# storage_layer.py
from __future__ import annotations

import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import orjson

from pydantic import BaseModel, Field
from loguru import logger

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

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
    upload_timeout_s: int = Field(default=120, ge=1, le=3600)


class BlobStorage:
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
            svc = BlobServiceClient.from_connection_string(cfg.connection_string)
        elif cfg.account_url:
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
        var: str,
        species: Union[int, str],
        time: Union[int, str],
        z: int,
        x: int,
        y: int,
        ext: str = "png",
    ) -> bytes:
        addr = TileAddress(
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
            return
        
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
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((p, e))
                    if fail_fast:
                        for f in futs:
                            f.cancel()
                        break
        
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
        Parallel upload of an on-disk tile tree to Blob-compatible storage.

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
            return

        workers = max_workers or self._cfg.max_workers

        def _upload_one(p: Path) -> None:
            rel = p.relative_to(root).as_posix()
            blob_name = f"{remote_root}/{rel}"
            ctype = _guess_content_type(p)
            cache = cache_control_pointer if rel.endswith("current.json") else cache_control_immutable

            data = p.read_bytes()
            self.store_bytes(
                blob_name=blob_name,
                data=data,
                content_type=ctype,
                cache_control=cache,
                overwrite=overwrite,
            )

        errors = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_upload_one, p): p for p in files}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((p, e))
                    if fail_fast:
                        # Cancel remaining work
                        for f in futs:
                            f.cancel()
                        break

        if errors:
            msg = "\n".join([f"- {p}: {type(e).__name__}: {e}" for p, e in errors[:20]])
            raise RuntimeError(f"store_tiles() failed for {len(errors)} file(s). First errors:\n{msg}")

####################################################################################################################
# Factory
####################################################################################################################

def create_storage(
    *,
    container: str = "viz",
    # If provided, takes precedence (good for Azurite)
    connection_string: Optional[str] = None,
    # For Azure MSI mode
    account_url: Optional[str] = None,
    # Upload tuning
    max_workers: int = 16,
    upload_timeout_s: int = 120,
    create_container_if_missing: bool = True,
) -> BlobStorage:
    """
    Create a BlobStorage instance for either Azurite or Azure.

    """
    cs = None
    au = None
    use_cred = False
    
    # Priority: explicit args > env vars
    if connection_string:
        cs = connection_string
    elif account_url:
        au = account_url
        use_cred = True
    else:
        # Try Azurite first
        cs = os.getenv("AZURITE_CONN_STR") if os.getenv("USE_AZURITE", "false").lower() == "true" else os.getenv("AZURE_CONN_STR")
        if not cs:
            # Fall back to Azure
            au = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            use_cred = True

    cfg = StorageConfig(
        container=container,
        connection_string=cs,
        account_url=au,
        use_default_azure_credential=use_cred,
        create_container_if_missing=create_container_if_missing,
        max_workers=max_workers,
        upload_timeout_s=upload_timeout_s,
    )
    return BlobStorage(cfg)
