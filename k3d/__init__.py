import json
from pathlib import Path
from typing import Dict as TypingDict
from typing import List as TypingList

from ._protocol import switch_to_binary_protocol, switch_to_text_protocol
from ._version import __version__
from .colormaps import (basic_color_maps, matplotlib_color_maps,
                        paraview_color_maps)
from .factory import (_default_color, default_colormap, label, line, lines,
                      marching_cubes, mesh, mip, nice_colors, plot, points,
                      sparse_voxels, stl, surface, text, text2d, texture,
                      texture_text, vector_field, vectors, volume,
                      volume_slice, voxel_chunk, voxels, voxels_group,
                      vtk_poly_data)
from .objects import clone_object, create_object
from .plot import Plot
from .transfer_function_editor import transfer_function_editor
from .transform import transform

HERE = Path(__file__).parent.resolve()

with (HERE / "labextension" / "package.json").open() as fid:
    data = json.load(fid)


def _jupyter_labextension_paths() -> TypingList[TypingDict[str, str]]:
    return [{"src": "labextension", "dest": data["name"]}]


def _jupyter_nbextension_paths() -> TypingList[TypingDict[str, str]]:
    return [
        {
            "section": "notebook",
            "src": "static",
            "dest": "k3d",
            "require": "k3d/extension",
        }
    ]
