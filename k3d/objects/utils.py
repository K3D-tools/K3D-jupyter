"""Utility functions and objects map for K3D objects."""

from typing import Any, Dict as TypingDict, Union

# Import all object classes
from .base import VoxelChunk
from .geometry import Line, Lines, Mesh, STL, Surface
from .points import Points
from .text import Text, Text2d, Label, TextureText
from .texture import Texture
from .vectors import VectorField, Vectors
from .volumetric import (
    MarchingCubes,
    Volume,
    VolumeSlice,
    Voxels,
    SparseVoxels,
    VoxelsGroup,
    MIP,
)
from .._version import __version__ as version

# Objects mapping for factory functions
objects_map: TypingDict[str, Any] = {
    'Line': Line,
    'Label': Label,
    'Lines': Lines,
    'MIP': MIP,
    'MarchingCubes': MarchingCubes,
    'Mesh': Mesh,
    'Points': Points,
    'STL': STL,
    'SparseVoxels': SparseVoxels,
    'Surface': Surface,
    'Text': Text,
    'Text2d': Text2d,
    'Texture': Texture,
    'TextureText': TextureText,
    'VectorField': VectorField,
    'Vectors': Vectors,
    'Volume': Volume,
    'VolumeSlice': VolumeSlice,
    'Voxels': Voxels,
    'VoxelsGroup': VoxelsGroup
}


def create_object(obj: TypingDict[str, Any], is_chunk: bool = False) -> Union[VoxelChunk, Any]:
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

    attributes = {
        k: from_json(obj[k]) for k in obj.keys() if k != 'type'
    }

    # force to use current version
    attributes['_model_module'] = 'k3d'
    attributes['_model_module_version'] = version
    attributes['_view_module_version'] = version

    if is_chunk:
        return VoxelChunk(**attributes)
    else:
        return objects_map[obj['type']](**attributes)


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
        if "sync" in v.metadata and k not in ['id', 'type']:
            param[k] = obj[k]

    return objects_map[obj['type']](**param)
