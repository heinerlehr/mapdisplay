"""
Tile builder: converts zarr data to Web Mercator tiles in webP format.
Optimizations: cached colormaps, pre-computed value ranges, parallel I/O, optional GPU acceleration.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from iconfig.iconfig import iConfig

# Optional GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class VariableConfig(BaseModel):
    """Configuration for a single variable to be tiled."""
    name: str
    colormap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    is_mask: bool = False  # If True, generates to mask/{species}/z/x/y.webp


class TileConfig(BaseModel):
    """Configuration for tile generation."""
    zarr_file: Path
    output_dir: Path
    var_name: str = Field(default="suitability")
    species_id: str = Field(default="mpyr")
    time_index: int = Field(default=0)
    colormap: str = Field(default="viridis")
    vmin: Optional[float] = Field(default=None)
    vmax: Optional[float] = Field(default=None)

class WebMercatorTiler:
    """Generate Web Mercator tiles from gridded data."""
    
    TILE_SIZE = 256  # Standard tile size in pixels

    def __init__(self):
        self.config = iConfig()
    
    @staticmethod
    def get_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
        """
        Get geographic bounds (lon_min, lat_min, lon_max, lat_max) for a tile.
        
        Args:
            z: Zoom level
            x: Tile column
            y: Tile row
            
        Returns:
            (lon_min, lat_min, lon_max, lat_max) in geographic coordinates
        """
        n = 2.0 ** z
        
        # Longitude
        lon_min = (x / n) * 360 - 180
        lon_max = ((x + 1) / n) * 360 - 180
        
        # Latitude (inverse Mercator)
        def tile_to_lat(tile_y):
            return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * tile_y / n))))
        
        lat_max = tile_to_lat(y)
        lat_min = tile_to_lat(y + 1)
        
        return lon_min, lat_min, lon_max, lat_max
    
    def get_zoom_levels(self,data_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Determine reasonable zoom levels for data shape.
        
        Args:
            data_shape: (lat_size, lon_size)
            
        Returns:
            (min_zoom, max_zoom)
        """
        lat_size, lon_size = data_shape

        # Min zoom covers the entire world
        min_zoom = self.config("tiler.min_zoom_levels", default=0)

        if (max_zoom := self.config("tiler.max_zoom_levels")) is not None:
            return min_zoom, max_zoom
        
        # Calculate based on data resolution
        # Higher max zoom if we have high resolution data
        if lon_size >= 3600:  # 0.1 degree resolution
            max_zoom = 8
        elif lon_size >= 1800:  # 0.2 degree resolution
            max_zoom = 7
        elif lon_size >= 900:  # 0.4 degree resolution
            max_zoom = 6
        else:
            max_zoom = 5
        
        return min_zoom, max_zoom


class DataRenderer:
    """Render gridded data as images with caching and optional GPU acceleration."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Args:
            use_gpu: Use GPU (CuPy) for resampling if available
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self._colormap_cache: Dict[str, plt.cm.ScalarMappable] = {}
        
        if self.use_gpu and not HAS_CUPY:
            logger.warning("GPU requested but CuPy not available. Falling back to CPU.")
    
    def get_colormap(self, colormap_name: str):
        """Get cached colormap."""
        if colormap_name not in self._colormap_cache:
            cmap = plt.get_cmap(colormap_name)
            self._colormap_cache[colormap_name] = cmap
        return self._colormap_cache[colormap_name]
    
    @staticmethod
    def normalize_data(
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        clip_outliers: bool = True,
    ) -> np.ndarray:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Input array
            vmin/vmax: Value range for normalization
            clip_outliers: Clip values outside vmin/vmax
            
        Returns:
            Normalized array in [0, 1]
        """
        data = np.nan_to_num(data, nan=0.0)
        data = data.astype(float)
        
        if vmin is None or vmax is None:
            # Use provided vmin/vmax if both given, else compute from data
            valid_mask = data > -999
            if valid_mask.any():
                if vmin is None:
                    vmin = np.percentile(data[valid_mask], 2)
                if vmax is None:
                    vmax = np.percentile(data[valid_mask], 98)
            else:
                vmin = vmin or 0.0
                vmax = vmax or 1.0
        
        # Normalize to [0, 1]
        if (vmax - vmin) > 0:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)
        
        if clip_outliers:
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def apply_colormap(
        self,
        data: np.ndarray,
        colormap_name: str = "viridis",
    ) -> np.ndarray:
        """
        Apply colormap to normalized data.
        
        Args:
            data: Normalized data in [0, 1]
            colormap_name: Matplotlib colormap name
            
        Returns:
            RGBA image array (height, width, 4) with values in [0, 255]
        """
        cmap = self.get_colormap(colormap_name)
        rgba = cmap(data)
        
        # Convert to uint8
        return (rgba * 255).astype(np.uint8)
    
    def resample_to_tile_size(self, data: np.ndarray, tile_size: int = 256) -> np.ndarray:
        """Resample data to tile size using GPU if available, else CPU."""
        current_shape = data.shape
        if current_shape == (tile_size, tile_size):
            return data
        
        zoom_factors = (
            tile_size / current_shape[0],
            tile_size / current_shape[1],
        )
        
        if self.use_gpu:
            return self._resample_gpu(data, zoom_factors, tile_size)
        else:
            return self._resample_cpu(data, zoom_factors)
    
    @staticmethod
    def _resample_cpu(data: np.ndarray, zoom_factors: Tuple[float, float]) -> np.ndarray:
        """Resample on CPU using scipy."""
        from scipy.ndimage import zoom
        return zoom(data, zoom_factors, order=0)
    
    @staticmethod
    def _resample_gpu(data: np.ndarray, zoom_factors: Tuple[float, float], tile_size: int) -> np.ndarray:
        """Resample on GPU using CuPy with nearest-neighbor interpolation."""
        try:
            data_gpu = cp.asarray(data, dtype=cp.float32)
            h, w = data.shape
            new_h, new_w = tile_size, tile_size
            
            # Create coordinate grids for output (target) size
            y_out = cp.arange(new_h, dtype=cp.float32)
            x_out = cp.arange(new_w, dtype=cp.float32)
            
            # Map output coordinates back to input coordinates
            y_indices = cp.floor(y_out / zoom_factors[0]).astype(cp.int32)
            x_indices = cp.floor(x_out / zoom_factors[1]).astype(cp.int32)
            
            # Clamp to valid range
            y_indices = cp.clip(y_indices, 0, h - 1)
            x_indices = cp.clip(x_indices, 0, w - 1)
            
            # Create 2D mesh grids for indexing
            yy, xx = cp.meshgrid(y_indices, x_indices, indexing='ij')
            
            # Index into the data
            resampled_gpu = data_gpu[yy, xx]
            
            return cp.asnumpy(resampled_gpu)
        except Exception as e:
            logger.warning(f"GPU resampling failed, falling back to CPU: {e}")
            from scipy.ndimage import zoom
            return zoom(data, zoom_factors, order=0)


class TileBuilder:
    """Main tile builder class."""
    
    def __init__(self, config: TileConfig, use_gpu: bool = False):
        self.config = config
        self.tiler = WebMercatorTiler()
        self.renderer = DataRenderer(use_gpu=use_gpu)
    
    def build(self) -> None:
        """Build all tiles from zarr file."""
        logger.info(f"Loading zarr file: {self.config.zarr_file}")
        ds = xr.open_zarr(self.config.zarr_file)
        
        # Get variable data
        if self.config.var_name not in ds.variables:
            raise ValueError(f"Variable {self.config.var_name} not found in zarr file")
        
        data_array = ds[self.config.var_name]
        
        # Extract time slice if needed
        if "time" in data_array.dims:
            data_array = data_array.isel(time=self.config.time_index)
        
        # Get data as numpy array
        data = data_array.values
        if len(data.shape) > 2:
            # If still 3D, take first slice
            data = data[0]
        
        logger.info(f"Data shape: {data.shape}")
        
        # Get zoom levels
        min_zoom, max_zoom = self.tiler.get_zoom_levels(data.shape)
        logger.info(f"Generating tiles from zoom {min_zoom} to {max_zoom}")
        
        # Generate tiles for each zoom level
        for z in range(min_zoom, max_zoom + 1):
            self._generate_tiles_for_zoom(data, z)
        
        logger.info("Tile generation complete")
    
    def _generate_tiles_for_zoom(self, data: np.ndarray, z: int) -> None:
        """Generate all tiles for a specific zoom level."""
        # Number of tiles at this zoom level
        n_tiles = 2 ** z
        
        logger.info(f"Generating {n_tiles}x{n_tiles} tiles for zoom {z}")
        
        for y in range(n_tiles):
            for x in range(n_tiles):
                self._generate_single_tile(data, z, x, y)
    
    def _generate_single_tile(
        self,
        data: np.ndarray,
        z: int,
        x: int,
        y: int,
    ) -> None:
        """Generate a single tile.
        
        Args:
            data: Full data array (lat, lon)
            z: Zoom level
            x: Tile column
            y: Tile row
        """
        # Get geographic bounds for this tile
        lon_min, lat_min, lon_max, lat_max = self.tiler.get_tile_bounds(z, x, y)
        
        # Extract data for this tile bounds
        # Assuming data is in (lat, lon) order with uniform grid
        lat_indices, lon_indices = self._get_data_indices(data.shape, lat_min, lat_max, lon_min, lon_max)
        
        if lat_indices.size == 0 or lon_indices.size == 0:
            # Empty tile, create blank
            tile_data = np.zeros((self.tiler.TILE_SIZE, self.tiler.TILE_SIZE, 3), dtype=np.uint8)
        else:
            tile_data = data[np.ix_(lat_indices, lon_indices)]
            
            # Resample to tile size
            tile_data = self._resample_to_tile_size(tile_data)
            
            # Normalize
            tile_data = self.renderer.normalize_data(
                tile_data,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
            )
            
            # Apply colormap
            tile_rgba = self.renderer.apply_colormap(tile_data, self.config.colormap)
            
            # Drop alpha channel for webP
            tile_data = tile_rgba[:, :, :3]
        
        # Save tile
        self._save_tile(tile_data, z, x, y)
    
    def _get_data_indices(
        self,
        data_shape: Tuple[int, int],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices into data array for geographic bounds."""
        lat_size, lon_size = data_shape
        
        # Assume regular grid: latitude from 90 to -90, longitude from -180 to 180
        # Calculate pixel indices corresponding to these bounds
        
        # Latitude: from 90 at index 0 to -90 at index lat_size-1
        lat_px_min = int((90 - lat_max) / 180 * lat_size)
        lat_px_max = int((90 - lat_min) / 180 * lat_size)
        
        # Longitude: from -180 at index 0 to 180 at index lon_size-1
        lon_px_min = int((lon_min + 180) / 360 * lon_size)
        lon_px_max = int((lon_max + 180) / 360 * lon_size)
        
        # Clamp to valid ranges
        lat_px_min = max(0, min(lat_size - 1, lat_px_min))
        lat_px_max = max(0, min(lat_size - 1, lat_px_max))
        lon_px_min = max(0, min(lon_size - 1, lon_px_min))
        lon_px_max = max(0, min(lon_size - 1, lon_px_max))
        
        # Ensure min < max
        if lat_px_min >= lat_px_max:
            lat_px_max = lat_px_min + 1
        if lon_px_min >= lon_px_max:
            lon_px_max = lon_px_min + 1
        
        lat_indices = np.arange(lat_px_min, lat_px_max)
        lon_indices = np.arange(lon_px_min, lon_px_max)
        
        return lat_indices, lon_indices
    
    def _resample_to_tile_size(self, data: np.ndarray) -> np.ndarray:
        """Resample data to tile size (CPU or GPU)."""
        return self.renderer.resample_to_tile_size(data, self.tiler.TILE_SIZE)
    
    def _save_tile(self, tile_data: np.ndarray, z: int, x: int, y: int) -> None:
        """Save tile as webP."""
        # Create output path: var/species/time/z/x/y.webp
        tile_path = (
            self.config.output_dir 
            / str(self.config.var_name)
            / str(self.config.species_id)
            / "mean"
            / str(z)
            / str(x)
        )
        tile_path.mkdir(parents=True, exist_ok=True)
        
        tile_file = tile_path / f"{y}.webp"
        
        # Convert to image and save
        img = Image.fromarray(tile_data.astype(np.uint8), mode="RGB")
        img.save(tile_file, "webp", quality=85)


def build_tiles_from_zarr(
    zarr_file: Path,
    output_dir: Path,
    var_name: str = "suitability",
    species_id: str = "mpyr",
    time_index: int = 0,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Build Web Mercator tiles from a zarr file.
    
    Args:
        zarr_file: Path to zarr file
        output_dir: Directory to save tiles
        var_name: Variable to tile
        species_id: Species identifier
        time_index: Time slice to use
        colormap: Matplotlib colormap
        vmin/vmax: Data value range for normalization
    """
    config = TileConfig(
        zarr_file=zarr_file,
        output_dir=output_dir,
        var_name=var_name,
        species_id=species_id,
        time_index=time_index,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    
    builder = TileBuilder(config)
    builder.build()


class BatchTileBuilder:
    """Build tiles for multiple variables in a single pass with optimizations."""
    
    def __init__(
        self,
        config: iConfig,
        zarr_file: Path,
        output_dir: Path,
        variables: List[VariableConfig],
        include_time_average: bool = True,
        species_dim: str = "species",
        time_dim: str = "time",
        latitude_dim: str = "latitude",
        longitude_dim: str = "longitude",
        use_gpu: bool = False,
        max_workers: int = 4,
    ):
        """
        Args:
            zarr_file: Path to zarr file
            output_dir: Root output directory for tiles
            variables: List of VariableConfig for variables to tile
            include_time_average: Whether to generate time-averaged tiles
            species_dim: Name of species dimension
            time_dim: Name of time dimension
            latitude_dim: Name of latitude dimension
            longitude_dim: Name of longitude dimension
            use_gpu: Use GPU acceleration for resampling
            max_workers: Number of parallel workers for I/O
            config: iConfig instance for configuration
        """
        self.config = config
        self.zarr_file = zarr_file
        self.output_dir = output_dir
        self.variables = variables
        self.include_time_average = include_time_average
        self.species_dim = species_dim
        self.time_dim = time_dim
        self.latitude_dim = latitude_dim
        self.longitude_dim = longitude_dim
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.ds = None
        self.tiler = WebMercatorTiler()
        self.renderer = DataRenderer(use_gpu=use_gpu)
    
    def build(self) -> None:
        """Build all tiles with parallel I/O."""
        logger.info(f"Loading zarr file: {self.zarr_file}")
        self.ds = xr.open_zarr(self.zarr_file)
        
        # Use thread pool for I/O
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.executor = executor
            self.pending_tiles = []
            
            # For each variable
            for var_config in self.variables:
                self._process_variable(var_config)
            
            # Wait for all pending I/O to complete
            self._wait_for_pending_tiles()
        
        logger.info("Batch tile generation complete")
    
    def _wait_for_pending_tiles(self) -> None:
        """Wait for all pending tile I/O operations to complete."""
        for future in self.pending_tiles:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Tile save failed: {e}")
        self.pending_tiles.clear()
    
    def _queue_tile_save(self, tile_data: np.ndarray, z: int, x: int, y: int, 
                        var_name: str, species_id: str, time_label: str, is_mask: bool = False) -> None:
        """Queue a tile for parallel saving."""
        future = self.executor.submit(
            self._save_tile,
            tile_data.copy(),  # Copy to avoid data race
            z, x, y,
            var_name, species_id, time_label, is_mask
        )
        self.pending_tiles.append(future)
    
    def _process_variable(self, var_config: VariableConfig) -> None:
        """Process a single variable with pre-computed vmin/vmax."""
        var_name = var_config.name
        
        if var_name not in self.ds.variables:
            logger.warning(f"Variable {var_name} not found in zarr file")
            return
        
        data_array = self.ds[var_name]
        logger.info(f"Processing variable: {var_name}")
        
        # PRE-COMPUTE vmin/vmax if not provided (optimization: do once per variable)
        computed_vmin = var_config.vmin
        computed_vmax = var_config.vmax
        
        if computed_vmin is None or computed_vmax is None:
            logger.info(f"  Pre-computing value range for {var_name}")
            # Get all valid data for percentile computation
            all_data = data_array.values
            all_data = np.nan_to_num(all_data, nan=0.0)
            all_data = all_data.astype(float)
            valid_mask = all_data > -999
            
            if valid_mask.any():
                if computed_vmin is None:
                    computed_vmin = np.percentile(all_data[valid_mask], 2)
                if computed_vmax is None:
                    computed_vmax = np.percentile(all_data[valid_mask], 98)
                logger.info(f"    vmin={computed_vmin:.4f}, vmax={computed_vmax:.4f}")
        
        # Detect dimensions
        has_species = self.species_dim in data_array.dims
        has_time = self.time_dim in data_array.dims
        
        # Get species list
        species_list = [None]
        if has_species:
            species_list = list(self.ds[self.species_dim].values)
        
        # Get timesteps
        timesteps = [None]
        if has_time:
            if (tr := self.config("tiler.time_range")):
                start_time, end_time = tr
                start_time = np.datetime64(start_time)
                end_time = np.datetime64(end_time)
                time_values = self.ds[self.time_dim].values
                timesteps = [i for i, t in enumerate(time_values) if start_time <= t <= end_time]
            else:
                timesteps = list(range(len(self.ds[self.time_dim])))
        
        # Process each species
        for species in species_list:
            species_id = species if species is not None else "all"
            
            # Extract data for this species ONCE (optimization)
            if has_species:
                species_data = data_array.sel({self.species_dim: species})
            else:
                species_data = data_array
            
            # Process each timestep
            for time_idx in timesteps:
                # Extract time slice
                if has_time:
                    time_data = species_data.isel({self.time_dim: time_idx})
                    time_label = str(time_idx)
                else:
                    time_data = species_data
                    time_label = "current"
                
                # Get lat/lon data
                if self.latitude_dim not in time_data.dims or self.longitude_dim not in time_data.dims:
                    logger.warning(f"Variable {var_name} missing {self.latitude_dim}/{self.longitude_dim}")
                    continue
                
                data = time_data.values
                data = np.nan_to_num(data, nan=0.0)
                
                # Generate tiles (pass pre-computed vmin/vmax)
                var_config_copy = var_config
                var_config_copy.vmin = computed_vmin
                var_config_copy.vmax = computed_vmax
                
                self._generate_tiles_for_data(
                    data=data,
                    var_name=var_name,
                    species_id=species_id,
                    time_label=time_label,
                    var_config=var_config_copy,
                    is_mask=var_config.is_mask,
                )
            
            # Generate time average if requested and data has time dimension
            if self.include_time_average and has_time:
                time_averaged = species_data.mean(dim=self.time_dim)
                data = time_averaged.values
                data = np.nan_to_num(data, nan=0.0)
                
                var_config_copy = var_config
                var_config_copy.vmin = computed_vmin
                var_config_copy.vmax = computed_vmax
                
                self._generate_tiles_for_data(
                    data=data,
                    var_name=var_name,
                    species_id=species_id,
                    time_label="mean",
                    var_config=var_config_copy,
                    is_mask=var_config.is_mask,
                )
    
    def _generate_tiles_for_data(
        self,
        data: np.ndarray,
        var_name: str,
        species_id: str,
        time_label: str,
        var_config: VariableConfig,
        is_mask: bool = False,
    ) -> None:
        """Generate tiles for a specific data array."""
        min_zoom, max_zoom = self.tiler.get_zoom_levels(data.shape)
        logger.info(f"  Generating tiles: {var_name}/{species_id}/{time_label} (z{min_zoom}-{max_zoom})")
        
        for z in range(min_zoom, max_zoom + 1):
            n_tiles = 2 ** z
            for y in range(n_tiles):
                for x in range(n_tiles):
                    self._generate_single_tile(
                        data=data,
                        z=z,
                        x=x,
                        y=y,
                        var_name=var_name,
                        species_id=species_id,
                        time_label=time_label,
                        var_config=var_config,
                        is_mask=is_mask,
                    )
    
    def _generate_single_tile(
        self,
        data: np.ndarray,
        z: int,
        x: int,
        y: int,
        var_name: str,
        species_id: str,
        time_label: str,
        var_config: VariableConfig,
        is_mask: bool = False,
    ) -> None:
        """Generate a single tile and queue for I/O."""
        # Get tile bounds
        lon_min, lat_min, lon_max, lat_max = self.tiler.get_tile_bounds(z, x, y)
        
        # Extract data indices
        lat_indices, lon_indices = self._get_data_indices(
            data.shape, lat_min, lat_max, lon_min, lon_max
        )
        
        if lat_indices.size == 0 or lon_indices.size == 0:
            tile_data = np.zeros((self.tiler.TILE_SIZE, self.tiler.TILE_SIZE, 3), dtype=np.uint8)
        else:
            tile_data = data[np.ix_(lat_indices, lon_indices)]
            
            # Resample (uses GPU if configured)
            tile_data = self.renderer.resample_to_tile_size(tile_data, self.tiler.TILE_SIZE)
            
            # Normalize (with pre-computed vmin/vmax)
            tile_data = self.renderer.normalize_data(
                tile_data,
                vmin=var_config.vmin,
                vmax=var_config.vmax,
            )
            
            # Apply colormap (cached)
            tile_rgba = self.renderer.apply_colormap(tile_data, var_config.colormap)
            tile_data = tile_rgba[:, :, :3]
        
        # Queue tile for parallel saving
        self._queue_tile_save(
            tile_data=tile_data,
            z=z,
            x=x,
            y=y,
            var_name=var_name,
            species_id=species_id,
            time_label=time_label,
            is_mask=is_mask,
        )
    
    def _get_data_indices(
        self,
        data_shape: Tuple[int, int],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices into data array for geographic bounds."""
        lat_size, lon_size = data_shape
        
        # Assume regular grid: latitude from 90 to -90, longitude from -180 to 180
        lat_px_min = int((90 - lat_max) / 180 * lat_size)
        lat_px_max = int((90 - lat_min) / 180 * lat_size)
        
        lon_px_min = int((lon_min + 180) / 360 * lon_size)
        lon_px_max = int((lon_max + 180) / 360 * lon_size)
        
        # Clamp to valid ranges
        lat_px_min = max(0, min(lat_size - 1, lat_px_min))
        lat_px_max = max(0, min(lat_size - 1, lat_px_max))
        lon_px_min = max(0, min(lon_size - 1, lon_px_min))
        lon_px_max = max(0, min(lon_size - 1, lon_px_max))
        
        # Ensure min < max
        if lat_px_min >= lat_px_max:
            lat_px_max = lat_px_min + 1
        if lon_px_min >= lon_px_max:
            lon_px_max = lon_px_min + 1
        
        lat_indices = np.arange(lat_px_min, lat_px_max)
        lon_indices = np.arange(lon_px_min, lon_px_max)
        
        return lat_indices, lon_indices
    
    def _save_tile(
        self,
        tile_data: np.ndarray,
        z: int,
        x: int,
        y: int,
        var_name: str,
        species_id: str,
        time_label: str,
        is_mask: bool = False,
    ) -> None:
        """Save tile as webP."""
        if is_mask:
            # Mask tiles: mask/{species}/{time}/z/x/y.webp
            tile_path = (
                self.output_dir
                / "mask"
                / str(species_id)
                / str(time_label)
                / str(z)
                / str(x)
            )
        else:
            # Data tiles: {var}/{species}/{time}/z/x/y.webp
            tile_path = (
                self.output_dir
                / str(var_name)
                / str(species_id)
                / str(time_label)
                / str(z)
                / str(x)
            )
        
        tile_path.mkdir(parents=True, exist_ok=True)
        tile_file = tile_path / f"{y}.webp"
        
        img = Image.fromarray(tile_data.astype(np.uint8), mode="RGB")
        img.save(tile_file, "webp", quality=85)


def build_tiles_batch(
    config: iConfig,
    zarr_file: Path,
    output_dir: Path,
    variables: List[VariableConfig],
    include_time_average: bool = True,
    species_dim: str = "species",
    time_dim: str = "time",
    latitude_dim: str = "latitude",
    longitude_dim: str = "longitude",
    use_gpu: bool = False,
    max_workers: int = 4,
) -> None:
    """
    Build Web Mercator tiles for multiple variables in a single pass with optimizations.
    
    Args:
        zarr_file: Path to zarr file
        output_dir: Root directory for tiles
        variables: List of VariableConfig for each variable to tile
        include_time_average: Whether to generate time-averaged tiles
        species_dim: Name of species dimension
        time_dim: Name of time dimension
        latitude_dim: Name of latitude dimension
        longitude_dim: Name of longitude dimension
        use_gpu: Use GPU acceleration (CuPy) for resampling if available
        max_workers: Number of parallel workers for I/O operations
    
    Example:
        variables = [
            VariableConfig(name="temperature", colormap="coolwarm"),
            VariableConfig(name="salinity", colormap="haline"),
            VariableConfig(name="land_mask", is_mask=True),
        ]
        
        build_tiles_batch(
            zarr_file=Path("ocean.zarr"),
            output_dir=Path("tiles"),
            variables=variables,
            include_time_average=True,
            use_gpu=True,
            max_workers=8,
        )
    """
    builder = BatchTileBuilder(
        config=config,
        zarr_file=zarr_file,
        output_dir=output_dir,
        variables=variables,
        include_time_average=include_time_average,
        species_dim=species_dim,
        time_dim=time_dim,
        latitude_dim=latitude_dim,
        longitude_dim=longitude_dim,
        use_gpu=use_gpu,
        max_workers=max_workers,
    )
    builder.build()
