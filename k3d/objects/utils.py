"""Utility functions and objects map for K3D objects."""

import math
from typing import Any
from typing import Dict as TypingDict
from typing import Union

# Import all object classes
from .base import VoxelChunk
from .geometry import STL, Line, Lines, Mesh, Surface
from .points import Points
from .text import Label, Text, Text2d, TextureText
from .texture import Texture
from .vectors import VectorField, Vectors
from .volumetric import (MIP, MarchingCubes, SparseVoxels, Volume, VolumeSlice,
                         Voxels, VoxelsGroup)

# Objects mapping for factory functions
objects_map: TypingDict[str, Any] = {
    "Line": Line,
    "Label": Label,
    "Lines": Lines,
    "MIP": MIP,
    "MarchingCubes": MarchingCubes,
    "Mesh": Mesh,
    "Points": Points,
    "STL": STL,
    "SparseVoxels": SparseVoxels,
    "Surface": Surface,
    "Text": Text,
    "Text2d": Text2d,
    "Texture": Texture,
    "TextureText": TextureText,
    "VectorField": VectorField,
    "Vectors": Vectors,
    "Volume": Volume,
    "VolumeSlice": VolumeSlice,
    "Voxels": Voxels,
    "VoxelsGroup": VoxelsGroup,
}


def create_object(
        obj: TypingDict[str, Any], is_chunk: bool = False
) -> Union[VoxelChunk, Any]:
    """Create an object from a dictionary representation.

    Parameters
    ----------
    obj : dict
        Dictionary containing object data.
    is_chunk : bool, optional
        Whether this is a voxel chunk object, by default False.

    Returns
    -------
    object
        The created object instance.
    """
    from ..helpers import from_json

    attributes = {k: from_json(obj[k]) for k in obj.keys() if k != "type"}

    # Snapshots written before 2.19.0 carry shininess. The trait itself is a tombstone
    # that raises on any value, so old files are translated here, at the file boundary.
    shininess = attributes.pop("shininess", None)
    if shininess is not None and "roughness" not in attributes:
        attributes["roughness"] = math.sqrt(2.0 / (max(float(shininess), 0.0) + 2.0))

    # widget wiring keys (old snapshots: _model_*/_view_*; new ones: _kind,
    # _synced_props) are transport details, not object state
    attributes = {k: v for k, v in attributes.items() if not k.startswith("_")}

    if is_chunk:
        return VoxelChunk(**attributes)
    else:
        return objects_map[obj["type"]](**attributes)


def clone_object(obj: Any) -> Any:
    """Clone an existing object.

    Parameters
    ----------
    obj : object
        The object to clone.

    Returns
    -------
    object
        A new instance of the same object type with copied attributes.
    """
    param: TypingDict[str, Any] = {}

    for k, v in obj.traits().items():
        if k in obj._synced_props and k not in ["id", "type"]:
            param[k] = obj[k]

    return objects_map[obj["type"]](**param)
