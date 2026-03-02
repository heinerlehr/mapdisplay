from pydantic import BaseModel, Field
from typing import List, Optional, Union

import numpy as np
import xarray as xr

class TileMeta(BaseModel):
    version: str = Field(default="v1")
    zmax: int
    continuous_vars: List[str]
    boolean_vars: List[str]
    mask_var: Optional[str]
    species: List[str]
    time_labels: List[str]
    has_avg: bool

class TileAddress(BaseModel):
    """
    Canonical tile address -> blob name.
    
    Layout for data tiles:
      {var}/{species}/{time}/{z}/{x}/{y}.png
    
    Layout for mask tiles:
      mask/{species}/{time}/{z}/{x}/{y}.png
    
    Examples:
      suitability/slat/5/4/8/5.png  (data)
      mask/slat/5/4/8/5.png         (mask)
    """
    version: str
    var: str  # e.g., "suitability", "temperature"; or "mask" for mask layers
    species: Union[int, str]  # e.g., "slat", 0, 1
    time: Union[int, str]  # 0..23 or "mean"/"avg"
    z: int
    x: int
    y: int
    ext: str = Field(default="png")

    def blob_name(self) -> str:
        """Generate blob path: {var}/{species}/{time}/{z}/{x}/{y}.{ext}"""
        return (
            f"tiles/{self.version}/{self.var}/{self.species}/{self.time}/{self.z}/{self.x}/{self.y}.{self.ext}"
        )
    
    def is_mask(self) -> bool:
        """Check if this is a mask tile."""
        return self.var == "mask"
  

class VariableConfig(BaseModel):
    """Configuration for a single variable to be tiled."""
    name: str
    colormap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    is_mask: bool = False  # If True, generates to mask/{species}/z/x/y.png
    transparent_values: Optional[list[float]] = None  # e.g., [0.0] or None for default
    exclude_zero_from_transparent: bool = False
    type: str = "numerical"  # "numerical" or "categorical"
    colormap: Optional[str] = "coolwarm"  # e.g., "viridis", "tab10"; only for categorical
    mean: Optional[str] = "mean"
    preprocess: Optional[str] = None  # e.g., "mask" to apply mask before tiling

    def process(self, ds: xr.Dataset, **kwargs) -> xr.DataArray:
        match self.preprocess:
            case "mask":
                if (mask_name := kwargs.get("mask_name")) is None:
                    if "mask" in kwargs:
                        mask = kwargs["mask"]
                    else:
                        raise ValueError("Mask variable name must be provided in kwargs as 'mask_name' when preprocess is 'mask'")
                else:
                    mask = ds[mask_name]

                return ds[self.name].where(mask == 1)
            case _:
                return ds[self.name]

    def timemean(self, data: xr.DataArray, dims: List|str) -> xr.DataArray:
        """Compute the time mean of the data based on the specified method."""
        match self.mean:
            case "mean":
                return self._timemean(data, dims)
            case "annual":
                return self._annual(data, time_dim="time")
            case "mode":
                return self._mode(data, time_dim="time")
            case _:
                raise ValueError(f"Unsupported timemean method: {self.mean}")

    def _timemean(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Compute the mean across all dimensions except latitude and longitude."""
        
        if time_dim not in data.dims:
            return data  # No time dimension, return as is
        
        # Compute mean across those dimensions
        return data.mean(dim=time_dim)
    
    def _annual(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Compute the annual mean by grouping by year and averaging."""
        if time_dim not in data.dims:
            return data  # No time dimension, return as is
        
        months = len(data[time_dim])
        data = data.sum(dim=time_dim)  # Sum over time dimension to get annual totals
        
        data = data / months * 12  # Scale to annual average (assuming monthly data)
        return data

    def _mode(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Compute the mode (most frequent value) across the time dimension."""
        if time_dim not in data.dims:
            return data  # No time dimension, return as is
        
        def find_mode(x):
            # Filter out NaN values
            x_clean = x[~np.isnan(x)]
            if len(x_clean) == 0:
                return np.nan
            values, counts = np.unique(x_clean, return_counts=True)
            mode = float(values[np.argmax(counts)])
            return mode

        mode = xr.apply_ufunc(find_mode, data, input_core_dims=[[time_dim]], vectorize=True)
        return mode
    
    def get_colormap(self, variable_name: str) -> str:
        """Get the colormap for a variable from the configuration."""
        variable_config = next((var for var in self.config("variables", default=[]) if var.get("name") == variable_name), None)
        if variable_config is None:
            raise ValueError(f"No configuration found for variable '{variable_name}'")
        return variable_config.get("colormap", "coolwarm")
    
    def scale(self, data: np.ndarray, min_value:float, max_value:float) -> np.ndarray:
        """Scale data to 0-255 based on vmin/vmax."""
        match self.type:
            case "categorical":
                return self._scale_categorical(data, self.colormap)
            case _:
                return self._scale_numerical(data, min_value, max_value)

    def _scale_categorical(self, data: np.ndarray, colormap_name: str) -> np.ndarray:
        """Scale categorical data to 0-255 based on unique values and colormap."""
        unique_values = np.unique(np.round(data,0))
        unique_values = unique_values[~np.isnan(unique_values)]
        n_categories = len(unique_values)
        if n_categories > 256:
            raise ValueError(f"Too many categories ({n_categories}) to fit in 8-bit color")
        
        # Create a mapping from category to color index
        category_to_index = {cat: idx for idx, cat in enumerate(unique_values)}
        indexed_data = np.empty_like(data, dtype=float)
        for key, value in category_to_index.items():
            indexed_data[data == key] = value
        
        # Optionally apply a colormap here if you want specific colors for categories
        # For now, we just return the indexed data as uint8
        return indexed_data/len(unique_values) if len(unique_values) > 1 else indexed_data

    def _scale_numerical(self, data: np.ndarray, min_value:int=0, max_value:int=255) -> np.ndarray:
        if min_value is None or max_value is None:
            raise ValueError("min_value and max_value must be set for scaling")
        scaled = (data - min_value) / (max_value - min_value) * (max_value - min_value) + min_value
        return np.clip(scaled, min_value, max_value)

class MapDefinition(BaseModel):
    version_id: str
    created_at: str
    mapfile: str
    variables: List[VariableConfig]
    mask: Optional[str]
    species: List[str]
    time_range: Optional[List[int]]
    time_labels: Optional[List[str]]
    zoom_levels: List[int]
    categorical_labels: Optional[dict[str, list]] = None  # Map variable name to category labels
    units: Optional[dict[str, str]] = None  # Map variable name to units

    def __contains__(self, item: str) -> bool:
        """Check if a variable is defined in the map definition."""
        return hasattr(self, item) and getattr(self, item) is not None
    
    def __getitem__(self, item: str):
        return getattr(self, item)
    
    def get(self, item: str, default=None):
        return getattr(self, item, default)