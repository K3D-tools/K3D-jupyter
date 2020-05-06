from ._version import version_info, __version__

from .colormaps import paraview_color_maps
from .colormaps import basic_color_maps
from .colormaps import matplotlib_color_maps

from .factory import (plot,
                      nice_colors,
                      line,
                      marching_cubes,
                      mesh,
                      points,
                      stl,
                      surface,
                      text,
                      text2d,
                      texture,
                      texture_text,
                      vector_field,
                      vectors,
                      voxels,
                      voxels_group,
                      sparse_voxels,
                      volume,
                      mip,
                      label,
                      vtk_poly_data,
                      voxel_chunk)

from .transfer_function_editor import transfer_function_editor

from .transform import transform


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'k3d',
        'require': 'k3d/extension'
    }]
