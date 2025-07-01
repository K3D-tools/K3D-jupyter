"""Factory package for K3D object creation functions."""

# Import all factory functions and constants from submodules
from .common import _default_color, nice_colors, default_colormap
from .geometry import line, lines, mesh, surface, stl
from .plot import plot
from .points import points
from .text import text, text2d, label, texture_text
from .texture import texture
from .vectors import vector_field, vectors
from .volumetric import volume, mip, volume_slice, voxels, sparse_voxels, voxels_group, \
    marching_cubes, voxel_chunk
from .vtk import vtk_poly_data

# Export all factory functions and constants
__all__ = [
    # Factory functions
    'line', 'lines', 'mesh', 'surface', 'stl',
    'volume', 'mip', 'volume_slice', 'voxels', 'sparse_voxels', 'voxels_group', 'marching_cubes',
    'voxel_chunk',
    'text', 'text2d', 'label', 'texture_text',
    'vector_field', 'vectors',
    'points',
    'texture',
    'vtk_poly_data',
    'plot',

    # Constants
    '_default_color', 'nice_colors', 'default_colormap',
]

# This file makes the factory directory a Python package
