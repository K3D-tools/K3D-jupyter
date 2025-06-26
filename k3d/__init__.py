import json
from pathlib import Path

from ._protocol import switch_to_binary_protocol, switch_to_text_protocol
from ._version import __version__
from .colormaps import basic_color_maps
from .colormaps import matplotlib_color_maps
from .colormaps import paraview_color_maps
from .factory import (plot,
                      nice_colors,
                      line,
                      lines,
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
                      volume_slice,
                      sparse_voxels,
                      volume,
                      mip,
                      label,
                      vtk_poly_data,
                      voxel_chunk)
from .objects import create_object, clone_object
from .plot import Plot
from .transfer_function_editor import transfer_function_editor
from .transform import transform

HERE = Path(__file__).parent.resolve()

with (HERE / "labextension" / "package.json").open() as fid:
    data = json.load(fid)


def _jupyter_labextension_paths():
    return [{
        "src": "labextension",
        "dest": data["name"]
    }]


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'k3d',
        'require': 'k3d/extension'
    }]
