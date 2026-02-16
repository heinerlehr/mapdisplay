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
      {var}/{species}/{time}/{z}/{x}/{y}.webp
    
    Layout for mask tiles:
      mask/{species}/{time}/{z}/{x}/{y}.webp
    
    Examples:
      suitability/slat/5/4/8/5.webp  (data)
      mask/slat/5/4/8/5.webp         (mask)
    """
    var: str  # e.g., "suitability", "temperature"; or "mask" for mask layers
    species: Union[int, str]  # e.g., "slat", 0, 1
    time: Union[int, str]  # 0..23 or "mean"/"avg"
    z: int
    x: int
    y: int
    ext: str = Field(default="webp")

    def blob_name(self) -> str:
        """Generate blob path: {var}/{species}/{time}/{z}/{x}/{y}.{ext}"""
        return (
            f"{self.var}/{self.species}/{self.time}/{self.z}/{self.x}/{self.y}.{self.ext}"
        )
    
    def is_mask(self) -> bool:
        """Check if this is a mask tile."""
        return self.var == "mask"