from enum import Enum, unique
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, validator, FilePath
import rmsp


@unique
class MeshFiletype(str, Enum):
    """Allowed system files types for mesh import."""

    DM = "dm"
    DXF = "dxf"
    STL = "stl"


class MeshInstance(BaseModel):
    """Mesh object and associated metadata."""

    label: str
    file: Union[FilePath, Tuple[FilePath, FilePath]]
    file_type: MeshFiletype
    is_solid: bool = False
    translation: Optional[Tuple[float, float, float]] = None
    mesh: Optional[Union[rmsp.Mesh, rmsp.Solid]] = None
    description: Optional[str] = None

    class Config:
        use_enum_values = True
        allow_mutation = True
        arbitrary_types_allowed = True

    @validator("file_type")
    def number_of_files(cls, v, values):
        if v == "dm":
            if not isinstance(values["file"], tuple):
                raise ValueError(
                    "File type 'dm' requires two files given in a tuple."
                )
        return v
