import os
from typing import Optional, Dict, List

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

from multiprocessing import Process

from loguru import logger
from iconfig.iconfig import iConfig

from md.model.version import Version
from md.storage.storage import create_storage
from md.preprocess.tile_builder import build_tiles_batch, VariableConfig

class CompletedProcess:
    """Fake Process object that appears to have already finished."""
    
    def __init__(self, pid: int = 0, exitcode: int = 0):
        self.pid = pid
        self.exitcode = exitcode
    
    def is_alive(self) -> bool:
        """Always False - process is already done."""
        return False
    
    def join(self, timeout: Optional[float] = None) -> None:
        """No-op - nothing to wait for."""
        pass
    
    def terminate(self) -> None:
        """No-op."""
        pass
    
    def kill(self) -> None:
        """No-op."""
        pass


def all_prepared(version: Version) -> bool:

    storage = create_storage()
    # Check that the version folder exists on the blob storage
    if not storage.has_version_folder(version=version.id):
        return False
    # Check that the mapfile file exists on the blob storage
    if not storage.has_mapfile(version=version.id, mapfile=version.mapfile):
        return False

    # Check that the zarr file exists on the blob storage
    zarr_file_name = version.mapfile.stem + ".zarr"
    if not storage.has_zarr_file(version=version.id, zarr_file=zarr_file_name):
        return False

    # Check that the tile tree exists on the blob storage
    if not storage.has_tile_subtree(var="suitability", species="mpyr", time="mean", z=0):
        return False

def preprocess(version: Version, background: bool = True) -> Process:
    if background:
        process = Process(target=run_preprocess, args=(version,))
        process.start()  # Start in background
    else:
        process = CompletedProcess()
        run_preprocess(version)  # Run in foreground
    return process

def run_preprocess(version: Version):
    # Load configuration
    config = iConfig()
    use_gpu = config.get("preprocess.use_gpu", default=False)
    max_workers = config.get("preprocess.max_workers", default=4)
    
    # Ensure mapfile exists locally
    mapfile_local_path = version.mapfile
    if not mapfile_local_path.exists():
        local_folder = Path(os.getenv('WOMAPP_PLUS_DATA'))
        mapfile_local_path = local_folder / version.mapfile
        if not mapfile_local_path.exists():
            logger.error(f"Mapfile {version.mapfile} not found locally at {mapfile_local_path}. Please ensure it is downloaded before preprocessing.")
            return
    
    # Create temp folder
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Copy mapfile to temp folder
        mapfile_name = mapfile_local_path.name
        temp_mapfile = temp_dir_path / f"{version.id}" / mapfile_name
        temp_mapfile.parent.mkdir(parents=True, exist_ok=True)  # Create {version.id}/ directory
        shutil.copy(mapfile_local_path, temp_mapfile)

        # Convert mapfile to zarr
        ds = prepare_ds(file=temp_mapfile)

        # Create tiles directory
        tiles_dir = temp_dir_path / f"{version.id}" / "tiles"

        # Run tile builder
        zarr_file = temp_mapfile.with_suffix('.zarr')
        run_tile_builder(
            config = config,
            zarr_file=zarr_file,
            output_dir=tiles_dir,
            ds=ds,
            use_gpu=use_gpu,
            max_workers=max_workers,
        )

        # Check all is in place
        if not temp_mapfile.exists() or not zarr_file.exists() or not tiles_dir.exists():
            logger.error("Preprocessing failed: missing expected output files.")
            return

        # Upload files to blob storage
        storage = create_storage()
        
        try:
            logger.info(f"Uploading mapfile to version {version.id}")
            storage.store_mapfile(version=version.id, mapfile_path=temp_mapfile)
            
            logger.info(f"Uploading zarr file to version {version.id}")
            storage.store_zarr(
                local_zarr_path=zarr_file,
                remote_prefix=f"{version.id}/{zarr_file.name}",
            )
            
            logger.info(f"Uploading tiles to version {version.id}")
            storage.store_tiles(
                local_tiles_root=tiles_dir,
                remote_root=f"tiles/{version.id}",
            )
            
            logger.info(f"Preprocessing complete for version {version.id}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

def prepare_ds(file: Path) -> xr.Dataset:
    """Prepare dataset from mapfile."""
    logger.info(f"Preparing dataset from {file}")
    ds = xr.open_dataset(file)
    
    # Save as zarr (data will be tiled during run_tile_builder)
    # Use zarr_v2 format to avoid unstable spec warnings for string types
    zarr_file = file.with_suffix('.zarr')
    ds.to_zarr(zarr_file, mode='w', zarr_format=2)
    logger.info(f"Zarr file saved to {zarr_file}")
    return ds


def get_mappable_variables(ds: xr.Dataset) -> Dict[str, Dict[str, bool]]:
    """
    Detect which variables can be mapped and their dimensions.
    
    Returns dict mapping variable names to dimension info:
        {"var_name": {"has_time": bool, "has_species": bool, "is_mask": bool}}
    """
    mappable = {}
    for var in ds.data_vars:
        # Must have latitude and longitude dimensions
        if "latitude" not in ds[var].dims or "longitude" not in ds[var].dims:
            continue
        
        has_time = "time" in ds[var].dims
        has_species = "species" in ds[var].dims
        is_mask = "mask" in var.lower()
        
        mappable[var] = {
            "has_time": has_time,
            "has_species": has_species,
            "is_mask": is_mask,
        }
    
    return mappable

def run_tile_builder(
    config: iConfig,
    zarr_file: Path,
    output_dir: Path,
    ds: xr.Dataset,
    use_gpu: bool = False,
    max_workers: int = 4,
) -> None:
    """Build Web Mercator tiles from zarr file."""
    logger.info(f"Building tiles from {zarr_file}")
    if use_gpu:
        logger.info("GPU acceleration enabled for resampling")
    
    # Get mappable variables
    mappable_vars = get_mappable_variables(ds)
    
    if not mappable_vars:
        logger.warning("No mappable variables found in dataset")
        return
    
    # Build variable configs
    variables: List[VariableConfig] = []
    if (vars_to_tile := config("tiler.vars", default=[])):
        # If specific vars are configured, filter to those
        mappable_vars = {var: info for var, info in mappable_vars.items() if var in vars_to_tile}
        if not mappable_vars:
            logger.warning(f"No configured variables found in dataset. Configured vars: {vars_to_tile}")
            return
    for var_name, var_info in mappable_vars.items():
        # Choose appropriate colormap for non-mask variables
        if var_info["is_mask"]:
            colormap = "gray"
        else:
            # Default colormaps based on common ocean variable names
            if "suitability" in var_name.lower():
                colormap = "coolwarm"
            elif "availability" in var_name.lower():
                colormap = "viridis"
            elif "limiting_factor" in var_name.lower():
                colormap = "viridis"
            elif "cultivate_species" in var_name.lower():
                colormap = "twilight"
            else:
                colormap = "viridis"  # Default
        
        variables.append(VariableConfig(
            name=var_name,
            colormap=colormap,
            is_mask=var_info["is_mask"],
        ))
    
    logger.info(f"Variables to tile: {[v.name for v in variables]}")
    
    # Build all tiles using batch API with optimization parameters
    build_tiles_batch(
        config=config,
        zarr_file=zarr_file,
        output_dir=output_dir,
        variables=variables,
        include_time_average=True,
        species_dim="species",
        time_dim="time",
        latitude_dim="latitude",
        longitude_dim="longitude",
        use_gpu=use_gpu,
        max_workers=max_workers,
    )
    
    logger.info(f"Tiles saved to {output_dir}")