"""K3D Objects Package.

This package contains all the drawable objects for K3D visualization.
"""

# Base classes and utilities
from .base import (
    Drawable,
    DrawableWithCallback,
    DrawableWithVoxelCallback,
    Group,
    VoxelChunk,
    TimeSeries,
    SingleOrList,
    ListOrArray,
    EPSILON,
)
# Geometric objects
from .geometry import (
    Line,
    Lines,
    Mesh,
    STL,
    Surface,
)
# Points objects
from .points import (
    Points,
)
# Text objects
from .text import (
    Text,
    Text2d,
    Label,
    TextureText,
)
# Texture objects
from .texture import (
    Texture,
)
# Utility functions
from .utils import (
    create_object,
    clone_object,
    objects_map,
)
# Vector objects
from .vectors import (
    VectorField,
    Vectors,
)
# Volumetric objects
from .volumetric import (
    MarchingCubes,
    Volume,
    VolumeSlice,
    Voxels,
    SparseVoxels,
    VoxelsGroup,
    MIP,
)

# Export all object classes for backward compatibility
__all__ = [
    # Base classes
    'Drawable',
    'DrawableWithCallback',
    'DrawableWithVoxelCallback',
    'Group',
    'VoxelChunk',
    'TimeSeries',
    'SingleOrList',
    'ListOrArray',
    'EPSILON',

    # Geometric objects
    'Line',
    'Lines',
    'Mesh',
    'STL',
    'Surface',

    # Volumetric objects
    'MarchingCubes',
    'Volume',
    'VolumeSlice',
    'Voxels',
    'SparseVoxels',
    'VoxelsGroup',
    'MIP',

    # Text objects
    'Text',
    'Text2d',
    'Label',
    'TextureText',

    # Vector objects
    'VectorField',
    'Vectors',

    # Points objects
    'Points',

    # Texture objects
    'Texture',

    # Utility functions
    'create_object',
    'clone_object',
    'objects_map',
]
