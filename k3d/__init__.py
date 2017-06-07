from ._version import version_info

from .colormaps.paraview_color_maps import paraview_color_maps
from .colormaps.basic_color_maps import basic_color_maps
from .colormaps.matplotlib_color_maps import matplotlib_color_maps

from .k3d import plot, points, line, vtk_poly_data, stl, texture, vectors, vector_fields, voxels, marching_cubes, mesh, \
    surface, text, text2d


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'k3d',
        'require': 'k3d/extension'
    }]
