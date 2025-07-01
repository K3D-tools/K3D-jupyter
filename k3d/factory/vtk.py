"""Factory function for VTK PolyData objects."""

import numpy as np
from typing import Union, List as TypingList, Optional, Dict as TypingDict, Any, Tuple

from .common import _default_color, default_colormap
from ..helpers import check_attribute_color_range
from ..objects import Mesh
from ..transform import process_transform_arguments

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]
ColorMap = Union[TypingList[TypingList[float]], TypingDict[str, Any], np.ndarray]
ColorRange = TypingList[float]
OpacityFunction = TypingList[float]

# Optional dependency
try:
    import vtk
    from vtk.util import numpy_support as nps
except ImportError:
    vtk = None
    nps = None


def vtk_poly_data(
        poly_data: Any,  # vtk.vtkPolyData
        color: int = _default_color,
        color_attribute: Optional[Tuple[str, float, float]] = None,
        color_map: Optional[ColorMap] = None,
        side: str = "front",
        slice_planes: ArrayLike = None,
        wireframe: bool = False,
        opacity: float = 1.0,
        volume: ArrayLike = None,
        volume_bounds: ArrayLike = None,
        opacity_function: OpacityFunction = None,
        color_range: ColorRange = None,
        cell_color_attribute: Optional[Tuple[str, float, float]] = None,
        flat_shading: bool = True,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Mesh:
    if slice_planes is None:
        slice_planes = []
    if volume is None:
        volume = []
    if volume_bounds is None:
        volume_bounds = []
    if opacity_function is None:
        opacity_function = []
    if color_range is None:
        color_range = []

    if color_map is None:
        color_map = default_colormap

    if vtk is None:
        raise RuntimeError("vtk module is not available")

    if (max(poly_data.GetPolys().GetMaxCellSize(), poly_data.GetStrips().GetMaxCellSize()) > 3):
        cut_triangles = vtk.vtkTriangleFilter()
        cut_triangles.SetInputData(poly_data)
        cut_triangles.Update()
        poly_data = cut_triangles.GetOutput()

    attribute = []
    triangles_attribute = []

    if color_attribute is not None:
        attribute = nps.vtk_to_numpy(
            poly_data.GetPointData().GetArray(color_attribute[0])
        )
        color_range = color_attribute[1:3]
    elif cell_color_attribute is not None:
        triangles_attribute = nps.vtk_to_numpy(
            poly_data.GetCellData().GetArray(cell_color_attribute[0])
        )
        color_range = cell_color_attribute[1:3]
    elif len(volume) > 0:
        color_range = check_attribute_color_range(volume, color_range)

    vertices = nps.vtk_to_numpy(poly_data.GetPoints().GetData())
    indices = nps.vtk_to_numpy(
        poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    volume_bounds = (
        np.array(volume_bounds, np.float32)
        if type(volume_bounds) is not dict
        else volume_bounds
    )

    return process_transform_arguments(
        Mesh(
            vertices=np.array(vertices, np.float32),
            indices=np.array(indices, np.uint32),
            normals=[],
            color=color,
            colors=[],
            opacity=opacity,
            attribute=np.array(attribute, np.float32),
            triangles_attribute=np.array(triangles_attribute, np.float32),
            color_range=color_range,
            color_map=np.array(color_map, np.float32),
            wireframe=wireframe,
            volume=volume,
            volume_bounds=volume_bounds,
            texture=None,
            opacity_function=opacity_function,
            side=side,
            flat_shading=flat_shading,
            slice_planes=slice_planes,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )
