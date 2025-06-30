# Re-export all factory functions from submodules
from .factory.common import _default_color, nice_colors, default_colormap
from .factory.geometry import line, lines, mesh, surface, stl
from .factory.volumetric import volume, mip, volume_slice, voxels, sparse_voxels, voxels_group, marching_cubes, voxel_chunk
from .factory.text import text, text2d, label, texture_text
from .factory.vectors import vector_field, vectors
from .factory.points import points
from .factory.texture import texture
from .factory.vtk import vtk_poly_data
from .factory.plot import plot

__all__ = [
    'line', 'lines', 'mesh', 'surface', 'stl',
    'volume', 'mip', 'volume_slice', 'voxels', 'sparse_voxels', 'voxels_group', 'marching_cubes',
    'text', 'text2d', 'label', 'texture_text',
    'vector_field', 'vectors',
    'points',
    'texture',
    'vtk_poly_data',
    'plot', 'voxel_chunk',
    '_default_color', 'nice_colors', 'default_colormap',
]
