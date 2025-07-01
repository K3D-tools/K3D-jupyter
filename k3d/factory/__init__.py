"""Factory package for K3D object creation functions."""

# Import all factory functions and constants from submodules
from .common import _default_color, default_colormap, nice_colors
from .geometry import line, lines, mesh, stl, surface
from .plot import plot
from .points import points
from .text import label, text, text2d, texture_text
from .texture import texture
from .vectors import vector_field, vectors
from .volumetric import (marching_cubes, mip, sparse_voxels, volume,
                         volume_slice, voxel_chunk, voxels, voxels_group)
from .vtk import vtk_poly_data

# Export all factory functions and constants
__all__ = [
    # Factory functions
    "line",
    "lines",
    "mesh",
    "surface",
    "stl",
    "volume",
    "mip",
    "volume_slice",
    "voxels",
    "sparse_voxels",
    "voxels_group",
    "marching_cubes",
    "voxel_chunk",
    "text",
    "text2d",
    "label",
    "texture_text",
    "vector_field",
    "vectors",
    "points",
    "texture",
    "vtk_poly_data",
    "plot",
    # Constants
    "_default_color",
    "nice_colors",
    "default_colormap",
]

# This file makes the factory directory a Python package
