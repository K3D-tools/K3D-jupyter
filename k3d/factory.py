# Re-export all factory functions from submodules
from .factory.common import _default_color, default_colormap, nice_colors
from .factory.geometry import line, lines, mesh, stl, surface
from .factory.plot import plot
from .factory.points import points
from .factory.text import label, text, text2d, texture_text
from .factory.texture import texture
from .factory.vectors import vector_field, vectors
from .factory.volumetric import (marching_cubes, mip, sparse_voxels, volume,
                                 volume_slice, voxel_chunk, voxels,
                                 voxels_group)
from .factory.vtk import vtk_poly_data

__all__ = [
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
    "voxel_chunk",
    "_default_color",
    "nice_colors",
    "default_colormap",
]
