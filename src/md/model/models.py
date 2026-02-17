from pydantic import BaseModel, Field
from typing import List, Optional, Union

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
    # NEW: specify which values should be transparent
    transparent_values: Optional[list[float]] = None  # e.g., [0.0] or None for default
    exclude_zero_from_transparent: bool = False


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

    def __contains__(self, item: str) -> bool:
        """Check if a variable is defined in the map definition."""
        return hasattr(self, item) and getattr(self, item) is not None
    
    def __getitem__(self, item: str):
        return getattr(self, item)
    
    def get(self, item: str, default=None):
        return getattr(self, item, default)