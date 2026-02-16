import os
from pathlib import Path
from typing import ClassVar, List, Optional
import orjson
from pydantic import BaseModel, Field, model_validator

import pandas as pd
import xarray as xr

from iconfig.iconfig import iConfig
from loguru import logger

from md.utils.utils import singleton, read_json


class Version(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    ID: ClassVar[str] = "id"
    CREATED_AT: ClassVar[str] = "created_at"
    MAPFILE: ClassVar[str] = "mapfile"

    id: str = Field(..., description="Version ID")
    description: str = Field(..., description="Version description")
    created_at: pd.Timestamp = Field(..., description="Version creation timestamp")
    mapfile: Path = Field(..., description="Path to the map file")
    annotations: Optional[List] = Field(default=None, description="List of annotations associated with this version")

    def __init__(self, **data):
        if self.CREATED_AT in data:
            data[self.CREATED_AT] = pd.to_datetime(data[self.CREATED_AT])
        if self.MAPFILE in data:
            data[self.MAPFILE] = Path(data[self.MAPFILE])

        super().__init__(**data)

    def __str__(self):
        return f"Version(id={self.id} (created_at={self.created_at}, mapfile={self.mapfile})"

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        versions = Versions()
        versions.update_version(self)
    
    def update_annotations(self, annotations: List|None=None, save: bool=True):
        t_annotations = []
        if self.mapfile.exists():
            ds = xr.open_dataset(self.mapfile)
            if "annotations" in ds:
                t_annotations = ds["annotations"].values.tolist()

        # Find if there are annotations with the same key
        if t_annotations:
            if annotations:
                for annotation in annotations:
                    for t_annotation in t_annotations:
                        if annotation["key"] == t_annotation["key"]:
                            t_annotation.update(annotation)
                            break
        else:
            t_annotations = annotations or []

        if t_annotations:
            self.annotations = t_annotations
            versions = Versions()
            versions.update_version(self, save=save)

@singleton
class Versions(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    versions: List[Version] = Field(default_factory=list, description="List of versions")

    def __init__(self, **data):
        super().__init__(**data)

    @model_validator(mode='after')
    def _load_versions(self):
        self._config = iConfig()

        if not (version_dir := Path(os.getenv("VERSION", "config"))).exists():
            logger.error(f"Version directory '{version_dir}' does not exist.")
            raise FileNotFoundError(f"Version directory '{version_dir}' does not exist.")
        self._version_file = version_dir / self._config("version_file", default="version.json")

        if not self._version_file.exists():
            logger.error(f"Version file '{self._version_file}' does not exist.")
            raise FileNotFoundError(f"Version file '{self._version_file}' does not exist.")
        data = read_json(self._version_file)
        self.versions = [Version(**row) for _, row in data.iterrows()]
        return self
    
    def save(self):
        data = [version.model_dump() for version in self.versions]
        with open(self._version_file, 'wb') as f:
            f.write(orjson.dumps(data))

    def get_version(self, id: str, **kwargs) -> Version:
        for version in self.versions:
            if version.id == id:
                return version
        
        raise ValueError(f"Version with id '{id}' not found.")

    def update_version(self, version: Version, save: bool=True):
        for idx, existing_version in enumerate(self.versions):
            if existing_version.id == version.id:
                self.versions[idx] = version
        else:
            self.versions.append(version)
        if save:
            self.save()
    
    def get_newest_version(self) -> Version:
        if not self.versions:
            logger.error("No versions available.")
            raise ValueError("No versions available.")
        return max(self.versions, key=lambda v: v.created_at)