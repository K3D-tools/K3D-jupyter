"""Utility functions and objects map for K3D objects."""

from .._version import __version__ as version

# Import all object classes
from .base import VoxelChunk
from .geometry import Line, Lines, Mesh, STL, Surface
from .volumetric import (
    MarchingCubes,
    Volume,
    VolumeSlice,
    Voxels,
    SparseVoxels,
    VoxelsGroup,
    MIP,
)
from .text import Text, Text2d, Label, TextureText
from .vectors import VectorField, Vectors
from .points import Points
from .texture import Texture


# Objects mapping for factory functions
objects_map = {
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


def create_object(obj, is_chunk=False):
    """Create an object from a dictionary representation.
    
    Args:
        obj: Dictionary containing object data
        is_chunk: Whether this is a voxel chunk object
        
    Returns:
        The created object instance
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


def clone_object(obj):
    """Clone an existing object.
    
    Args:
        obj: The object to clone
        
    Returns:
        A new instance of the same object type with copied attributes
    """
    param = {}

    for k, v in obj.traits().items():
        if "sync" in v.metadata and k not in ['id', 'type']:
            param[k] = obj[k]

    return objects_map[obj['type']](**param) 