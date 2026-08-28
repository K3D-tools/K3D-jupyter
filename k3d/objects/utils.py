"""Utility functions and objects map for K3D objects."""

import math
from typing import Any, Union
from typing import Dict as TypingDict

# Import all object classes
from .base import VoxelChunk
from .geometry import STL, Line, Lines, Mesh, Surface
from .points import Points
from .text import Label, Text, Text2d, TextureText
from .texture import Texture
from .vectors import VectorField, Vectors
from .volumetric import MIP, MarchingCubes, SparseVoxels, Volume, VolumeSlice, Voxels, VoxelsGroup

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

    attributes = {k: from_json(obj[k]) for k in obj if k != "type"}

    # Snapshots written before 3.0.0 carry shininess. The trait itself is a tombstone
    # that raises on any value, so old files are translated here, at the file boundary.
    shininess = attributes.pop("shininess", None)
    if shininess is not None and "roughness" not in attributes:
        attributes["roughness"] = math.sqrt(2.0 / (max(float(shininess), 0.0) + 2.0))

    # Same boundary for the pre-2.19 points shader alias: specular folded into '3d',
    # the highlights are driven by roughness/metalness.
    if attributes.get("shader") == "3dSpecular":
        attributes["shader"] = "3d"

    # widget wiring keys (old snapshots: _model_*/_view_*; new ones: _kind,
    # _synced_props) are transport details, not object state
    attributes = {k: v for k, v in attributes.items() if not k.startswith("_")}

    if is_chunk:
        return VoxelChunk(**attributes)
    return objects_map[obj["type"]](**attributes)


