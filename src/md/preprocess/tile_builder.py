"""Tile builder: converts zarr data to Web Mercator tiles in PNG format.
Optimizations: cached colormaps, pre-computed value ranges, parallel I/O, optional GPU acceleration.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from iconfig.iconfig import iConfig

from md.model.models import VariableConfig

# Optional GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

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
    
    def get_zoom_levels(self, data_shape: Tuple[int, int], calculate: bool=False) -> Tuple[int, int]:
        """
        Determine reasonable zoom levels for data shape.
        
        Args:
            data_shape: (lat_size, lon_size)
            
        Returns:
            (min_zoom, max_zoom)
        """
        _, lon_size = data_shape
        min_zoom = 0

        if not calculate:
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
        Normalize data to [0, 1] range, preserving NaN values for transparency.
        
        Args:
            data: Input array (may contain NaN values for transparency)
            vmin/vmax: Value range for normalization
            clip_outliers: Clip values outside vmin/vmax
            
        Returns:
            Normalized array in [0, 1] with NaN preserved
        """
        data = data.astype(float)
        
        # Preserve NaN mask - don't convert to 0
        nan_mask = np.isnan(data)
        data_for_stats = np.nan_to_num(data, nan=-999.0)  # Only for statistics
        
        if vmin is None or vmax is None:
            # Use provided vmin/vmax if both given, else compute from data (excluding NaNs)
            valid_mask = data_for_stats > -999
            if valid_mask.any():
                if vmin is None:
                    vmin = float(np.percentile(data_for_stats[valid_mask], 2))
                if vmax is None:
                    vmax = float(np.percentile(data_for_stats[valid_mask], 98))
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
        
        # Restore NaN values after normalization
        normalized[nan_mask] = np.nan
        
        return normalized
    
    def apply_colormap(
        self,
        data: np.ndarray,
        colormap_name: str = "viridis",
        transparent_values: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Apply colormap to normalized data.
        
        Args:
            data: Normalized data in [0, 1]
            colormap_name: Matplotlib colormap name
            transparent_values: List of values to be made transparent (alpha=0)
        Returns:
            RGBA image array (height, width, 4) with values in [0, 255]
        """
        cmap = self.get_colormap(colormap_name)
        rgba = cmap(data)
        
        # Convert to uint8
        rgba = (rgba * 255).astype(np.uint8)
        
        # Handle NaN values as transparent (set alpha to 0)
        nan_mask = np.isnan(data)
        rgba[nan_mask, 3] = 0
        
        return rgba
    
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
        # Store actual data bounds for coordinate mapping
        self.lat_min_data = None
        self.lat_max_data = None
        self.lon_min_data = None
        self.lon_max_data = None
        # Pixel resolution for accurate bounds adjustment
        self.lat_resolution = None
        self.lon_resolution = None
    
    def get_variables(self) -> List[VariableConfig]:
        """Return the list of variables with pre-computed vmin/vmax after building."""
        return self.variables
    
    def build(self) -> None:
        """Build all tiles with parallel I/O."""
        logger.info(f"Loading zarr file: {self.zarr_file}")
        self.ds = xr.open_zarr(self.zarr_file)
        
        # Read bounds from config, calculate resolution from actual data dimensions
        lat_config = self.config("tiler.copernicus_latitude")
        lon_config = self.config("tiler.copernicus_longitude")
        
        if lat_config is None or lon_config is None:
            logger.warning("Missing tiler config (copernicus_latitude/longitude), using dataset values")
            self.lat_min_data = float(self.ds[self.latitude_dim].min())
            self.lat_max_data = float(self.ds[self.latitude_dim].max())
            self.lon_min_data = float(self.ds[self.longitude_dim].min())
            self.lon_max_data = float(self.ds[self.longitude_dim].max())
        else:
            self.lat_min_data = float(lat_config["start"])
            self.lat_max_data = float(lat_config["end"])
            self.lon_min_data = float(lon_config["start"])
            self.lon_max_data = float(lon_config["end"])
        
        # Calculate resolution from actual data dimensions (more accurate than config)
        lat_size = len(self.ds[self.latitude_dim])
        lon_size = len(self.ds[self.longitude_dim])
        self.lat_resolution = (self.lat_max_data - self.lat_min_data) / (lat_size - 1) if lat_size > 1 else 1.0
        self.lon_resolution = (self.lon_max_data - self.lon_min_data) / (lon_size - 1) if lon_size > 1 else 1.0

        # Detect coordinate orientation from the dataset (robust against ascending/descending coords)
        lat_vals = np.asarray(self.ds[self.latitude_dim].values)
        lon_vals = np.asarray(self.ds[self.longitude_dim].values)

        self.lat_ascending = bool(lat_vals[-1] > lat_vals[0])  # True: south->north
        self.lon_ascending = bool(lon_vals[-1] > lon_vals[0])  # True: west->east (usually)

        # Detect whether the dataset uses 0..360 longitudes
        self.lon_0_360 = (self.lon_min_data >= 0.0 and self.lon_max_data > 180.0)

        # Set vmin and vmax
        for var_config in self.variables:
            var_name = var_config.name
            if var_name not in self.ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset, skipping")
                continue
            
            data_array = self.ds[var_name]
            logger.info(f"Processing variable: {var_name}")
            
            # PRE-COMPUTE vmin/vmax if not provided (optimization: do once per variable)
            computed_vmin = var_config.vmin
            computed_vmax = var_config.vmax

            if var_config.is_mask:
                computed_vmin = 0.0
                computed_vmax = 1.0            
            elif computed_vmin is None or computed_vmax is None:
                logger.info(f"  Pre-computing value range for {var_name}")
                all_data = data_array.values
                all_data = np.nan_to_num(all_data, nan=0.0)
                all_data = all_data.astype(float)
                valid_mask = all_data > -999
                
                if valid_mask.any():
                    if computed_vmin is None:
                        computed_vmin = float(np.min(all_data[valid_mask]))
                    if computed_vmax is None:
                        computed_vmax = float(np.max(all_data[valid_mask]))
                    logger.info(f"    vmin={computed_vmin:.4f}, vmax={computed_vmax:.4f}")
            
            # Update the variable config with pre-computed vmin/vmax for use in tile generation
            var_config.vmin = computed_vmin
            var_config.vmax = computed_vmax
        
        logger.info(f"Data bounds: lat [{self.lat_min_data:.2f}, {self.lat_max_data:.2f}], lon [{self.lon_min_data:.2f}, {self.lon_max_data:.2f}]")
        logger.info(f"Data dimensions: {lat_size} × {lon_size}")
        logger.info(f"Calculated resolution: lat {self.lat_resolution:.4f}°, lon {self.lon_resolution:.4f}°")
        
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
                var_config_copy = var_config.copy(update={"vmin": computed_vmin, "vmax": computed_vmax})
                
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
                
                var_config_copy = var_config.copy(update={"vmin": computed_vmin, "vmax": computed_vmax})
                
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
    def _tile_lonlat_grid(self, z: int, x: int, y: int, tile_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """
        Return lon/lat for pixel centers of a WebMercator XYZ tile.

        Output shapes: (tile_size, tile_size)
        Row 0 is the NORTH edge of the tile (correct for Leaflet).
        """
        n = 2.0 ** z

        # pixel centers in [0,1] within the tile
        u = (np.arange(tile_size, dtype=np.float64) + 0.5) / tile_size
        v = (np.arange(tile_size, dtype=np.float64) + 0.5) / tile_size
        uu, vv = np.meshgrid(u, v)  # (rows, cols) = (y, x)

        xtile = x + uu
        ytile = y + vv

        lon = (xtile / n) * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ytile / n))))
        return lon, lat

    def _normalize_lon_to_dataset(self, lon: np.ndarray) -> np.ndarray:
        """Map lon values to dataset convention (-180..180 or 0..360)."""
        if self.lon_0_360:
            # map to [0, 360)
            return np.mod(lon, 360.0)
        # map to (-180, 180]
        lon2 = (lon + 180.0) % 360.0 - 180.0
        return lon2


    def _sample_regular_latlon_grid_bilinear(
        self,
        data: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> np.ndarray:
        """
        Bilinear sampling from a regular lat/lon grid stored as a 2D numpy array [lat, lon].

        - data shape: (n_lat, n_lon)
        - lon/lat shapes: (H, W) (e.g., 256x256)
        - Returns: (H, W) float array, NaN outside bounds.
        - Handles lat/lon axis direction via self.lat_ascending/self.lon_ascending.
        - Honors dataset lon convention (-180..180 vs 0..360) via self._normalize_lon_to_dataset().
        - Treats NaNs in source data conservatively: if any of the 4 neighbors is NaN, output NaN.
        (You can relax this if you want "nan-aware" blending.)
        """
        n_lat, n_lon = data.shape

        # Normalize lon to dataset convention before indexing
        lon = self._normalize_lon_to_dataset(lon)

        # Geographic in-bounds mask
        lat_lo = min(self.lat_min_data, self.lat_max_data)
        lat_hi = max(self.lat_min_data, self.lat_max_data)

        if self.lon_0_360:
            lon_lo, lon_hi = self.lon_min_data, self.lon_max_data
            in_lon = (lon >= lon_lo) & (lon <= lon_hi)
        else:
            lon_lo, lon_hi = self.lon_min_data, self.lon_max_data
            in_lon = (lon >= lon_lo) & (lon <= lon_hi)

        in_lat = (lat >= lat_lo) & (lat <= lat_hi)
        in_bounds = in_lat & in_lon

        # Convert (lat,lon) -> fractional indices
        if self.lat_ascending:
            fy = (lat - self.lat_min_data) / self.lat_resolution
        else:
            fy = (self.lat_max_data - lat) / self.lat_resolution

        if self.lon_ascending:
            fx = (lon - self.lon_min_data) / self.lon_resolution
        else:
            fx = (self.lon_max_data - lon) / self.lon_resolution

        # For bilinear, we need floor indices and +1 neighbor
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Clamp to valid range for indexing (we'll NaN-out invalid via in_bounds + edge mask)
        x0c = np.clip(x0, 0, n_lon - 1)
        x1c = np.clip(x1, 0, n_lon - 1)
        y0c = np.clip(y0, 0, n_lat - 1)
        y1c = np.clip(y1, 0, n_lat - 1)

        # Fractional parts
        wx = fx - x0
        wy = fy - y0
        wx = wx.astype(np.float64)
        wy = wy.astype(np.float64)

        # Gather 4 neighbors
        v00 = data[y0c, x0c].astype(np.float64)
        v10 = data[y0c, x1c].astype(np.float64)
        v01 = data[y1c, x0c].astype(np.float64)
        v11 = data[y1c, x1c].astype(np.float64)

        # If any neighbor is NaN, output NaN (conservative)
        any_nan = np.isnan(v00) | np.isnan(v10) | np.isnan(v01) | np.isnan(v11)

        # Bilinear blend
        # v = (1-wy)*((1-wx)*v00 + wx*v10) + wy*((1-wx)*v01 + wx*v11)
        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        out = v0 * (1.0 - wy) + v1 * wy

        # Edge handling: if fx/fy land on last row/col, x1/y1 clamp collapses interpolation.
        # That’s fine, but we should avoid using pixels that conceptually require neighbors outside.
        # So we mark those as out-of-bounds too.
        needs_outside = (x0 < 0) | (y0 < 0) | (x1 >= n_lon) | (y1 >= n_lat)

        out[~in_bounds] = np.nan
        out[needs_outside] = np.nan
        out[any_nan] = np.nan

        return out


    def _sample_regular_latlon_grid_nearest(
        self,
        data: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> np.ndarray:
        """
        Nearest-neighbor sampling from a regular lat/lon grid stored as a 2D numpy array [lat, lon].

        Assumes the underlying grid spacing is constant and given by self.lat_resolution/self.lon_resolution.
        Handles lat/lon axis direction via self.lat_ascending/self.lon_ascending.
        Produces a (256,256) float array with NaN outside the dataset bounds.
        """
        lat_size, lon_size = data.shape

        # Normalize lon to dataset convention before indexing
        lon = self._normalize_lon_to_dataset(lon)

        # Build an in-bounds mask in geographic space
        lat_lo = min(self.lat_min_data, self.lat_max_data)
        lat_hi = max(self.lat_min_data, self.lat_max_data)

        if self.lon_0_360:
            lon_lo, lon_hi = self.lon_min_data, self.lon_max_data
            in_lon = (lon >= lon_lo) & (lon <= lon_hi)
        else:
            # typical -180..180 (no wrap assumed in your config)
            lon_lo, lon_hi = self.lon_min_data, self.lon_max_data
            in_lon = (lon >= lon_lo) & (lon <= lon_hi)

        in_lat = (lat >= lat_lo) & (lat <= lat_hi)
        in_bounds = in_lat & in_lon

        # Convert (lat,lon) -> fractional indices
        if self.lat_ascending:
            # index 0 at lat_min (south), increasing index goes north
            iy = (lat - self.lat_min_data) / self.lat_resolution
        else:
            # index 0 at lat_max (north), increasing index goes south
            iy = (self.lat_max_data - lat) / self.lat_resolution

        if self.lon_ascending:
            ix = (lon - self.lon_min_data) / self.lon_resolution
        else:
            ix = (self.lon_max_data - lon) / self.lon_resolution

        # Nearest integer indices
        iy = np.rint(iy).astype(np.int32)
        ix = np.rint(ix).astype(np.int32)

        # Clamp, but we will NaN out-of-bounds using in_bounds anyway
        iy = np.clip(iy, 0, lat_size - 1)
        ix = np.clip(ix, 0, lon_size - 1)

        sampled = data[iy, ix].astype(float)

        # Outside dataset bounds => transparent later (NaN)
        sampled[~in_bounds] = np.nan
        return sampled

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
        """Generate a single tile and queue for I/O, aligned to Leaflet XYZ (EPSG:3857)."""

        # 1) Compute lon/lat for each pixel center in the tile (WebMercator geometry)
        lon2d, lat2d = self._tile_lonlat_grid(z=z, x=x, y=y, tile_size=self.tiler.TILE_SIZE)

        # Convert to float for interpolation (masks are boolean)
        data = data.astype(float)

        # 2) Sample the source grid on that geometry (bilinear interpolation)
        tile_vals = self._sample_regular_latlon_grid_bilinear(data=data, lon=lon2d, lat=lat2d)

        # 3) Identify transparent pixels BEFORE converting to NaN (needed for step 5)
        transparent_mask = None
        if not is_mask and var_config.transparent_values:
            transparent_normalized = [float(v) for v in var_config.transparent_values]
            transparent_mask = np.isin(tile_vals.astype(float), transparent_normalized)
            if np.any(transparent_mask):
                tile_vals = np.where(transparent_mask, np.nan, tile_vals)

        # 4) Handle mask tiles differently: use values as alpha channel
        if is_mask:
            # For mask layers: normalize mask values to [0, 1] then invert for alpha
            # mask=0 -> alpha=255 (opaque, blocks data layer, shows base layer)
            # mask=1 -> alpha=0 (transparent, shows data layer through)

            tile_norm = self.renderer.normalize_data(
                tile_vals,
                vmin=var_config.vmin if var_config.vmin is not None else 0,
                vmax=var_config.vmax if var_config.vmax is not None else 1,
            )
            # Create RGBA with white RGB and inverted alpha (1 - mask)
            h, w = tile_norm.shape
            tile_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            tile_rgba[:, :, :3] = 255  # White RGB
            # Alpha channel: inverted normalized mask * 255 (0-255)
            # Where mask=1, alpha=0 (transparent); where mask=0, alpha=255 (opaque white)
            tile_rgba[:, :, 3] = np.nan_to_num((1 - tile_norm) * 255, nan=255).astype(np.uint8)
        else:
            # Normal data tiles: apply colormap (NaNs become alpha=0)
            tile_norm = self.renderer.normalize_data(
                tile_vals,
                vmin=var_config.vmin,
                vmax=var_config.vmax,
            )
            tile_rgba = self.renderer.apply_colormap(data=tile_norm, colormap_name=var_config.colormap)

            # 5) For data layers: explicitly set alpha=0 where transparent values occur
            if transparent_mask is not None:
                tile_rgba[transparent_mask, 3] = 0  # Set alpha channel to fully transparent

        # 6) Queue tile for parallel saving
        self._queue_tile_save(
            tile_data=tile_rgba,
            z=z,
            x=x,
            y=y,
            var_name=var_name,
            species_id=species_id,
            time_label=time_label,
            is_mask=is_mask,
        )
    
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
        """Save tile as PNG with RGBA support for transparency.
        
        PNG is used instead of WebP because Leaflet/Folium has better native support
        for PNG transparency in tile layers.
        
        Note: Tile data must be flipped vertically since zarr data is stored south-to-north
        but tiles require Row 0 (top) = North latitude.
        """
        if is_mask:
            # Mask tiles: mask/{species}/{time}/z/x/y.png
            tile_path = (
                self.output_dir
                / "mask"
                / str(species_id)
                / str(time_label)
                / str(z)
                / str(x)
            )
        else:
            # Data tiles: {var}/{species}/{time}/z/x/y.png
            tile_path = (
                self.output_dir
                / str(var_name)
                / str(species_id)
                / str(time_label)
                / str(z)
                / str(x)
            )
        
        tile_path.mkdir(parents=True, exist_ok=True)
        tile_file = tile_path / f"{y}.png"
        
        # Convert to image and save with proper mode for transparency
        # NOTE: Do NOT flip here. tile_rgba is already oriented with row 0 = North (Leaflet XYZ).
        mode = "RGBA" if tile_data.shape[2] == 4 else "RGB"
        img = Image.fromarray(tile_data.astype(np.uint8), mode=mode)
        # Use compress_level=6 for speed+size balance: ~10% CPU savings vs level 9 with minimal file size penalty
        img.save(tile_file, "png", compress_level=6)


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
    return builder.get_variables()
