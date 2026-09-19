"""Factory function for VTK PolyData objects."""

from typing import Any, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

from ..helpers import check_attribute_color_range
from ..objects import Mesh
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap

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
        roughness: float = 0.4,
        metalness: float = 0.0,
        shininess: float = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any,
) -> Mesh:
    """
    Create a Mesh drawable from a vtkPolyData object.

    Parameters
    ----------
    poly_data : vtkPolyData
        The polygonal data to convert. Cells with more than three points, and triangle strips, are
        triangulated first.
    color : int, optional
        Packed RGB color of the mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        Default is 255.
    color_attribute : tuple, optional
        Attribute to colour by, as (array name, min, max) read from the point data. Default is
        None.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    side : str, optional
        Which faces of the mesh are drawn: 'front', 'back' or 'double'. Default is 'front'.
    slice_planes : list, optional
        Planes [a, b, c, d] the section outline is drawn along, up to eight of them. The outline
        is drawn in the object colour. Default is None.
    wireframe : bool, optional
        Whether mesh should display as wireframe. Default is False.
    opacity : float, optional
        Opacity of mesh. Default is 1.0.
    volume : array_like, optional
        3D array sampled for the colour of each fragment, with volume_bounds giving the box it
        spans. Default is None.
    volume_bounds : array_like, optional
        Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] of `volume`. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    cell_color_attribute : tuple, optional
        Attribute to colour by, as (array name, min, max) read from the cell data. Default is
        None.
    flat_shading : bool, optional
        Whether mesh should display with flat shading. Default is True.
    roughness : float, optional
        Roughness of object material. Default is 0.4.
    metalness : float, optional
        Metalness of object material. Default is 0.0.
    shininess : float, optional
        Removed in 3.0.0; passing it raises. Use roughness and metalness. Default is None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Mesh
        The created Mesh object.
    """
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

    # indices below read GetPolys() only, so strips have to be triangulated whatever their
    # size: a strip of exactly 3 points left the mesh with no indices at all
    if (
            max(
                poly_data.GetPolys().GetMaxCellSize(),
                poly_data.GetStrips().GetMaxCellSize(),
            )
            > 3
            or poly_data.GetStrips().GetNumberOfCells() > 0
    ):
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
    indices = nps.vtk_to_numpy(poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
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
            roughness=roughness,
            metalness=metalness,
            shininess=shininess,
            slice_planes=slice_planes,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs,
    )
