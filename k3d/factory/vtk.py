"""Factory function for VTK PolyData objects."""

import numpy as np
from ..helpers import check_attribute_color_range
from ..objects import Mesh
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap

# Optional dependency
try:
    import vtk
    from vtk.util import numpy_support as nps
except ImportError:
    vtk = None
    nps = None


def vtk_poly_data(
        poly_data,
        color=_default_color,
        color_attribute=None,
        color_map=None,
        side="front",
        slice_planes=[],
        wireframe=False,
        opacity=1.0,
        volume=[],
        volume_bounds=[],
        opacity_function=[],
        color_range=[],
        cell_color_attribute=None,
        flat_shading=True,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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