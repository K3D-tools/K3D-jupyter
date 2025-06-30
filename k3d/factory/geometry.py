"""Factory functions for geometric objects."""

import numpy as np
import six

from ..helpers import check_attribute_color_range
from ..objects import Line, Lines, Mesh, STL, Surface
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap


def lines(
        vertices,
        indices,
        indices_type="triangle",
        color=_default_color,
        colors=[],  # lgtm [py/similar-function]
        attribute=[],
        color_map=None,
        color_range=[],
        width=0.01,
        shader="thick",
        shininess=50.0,
        radial_segments=8,
        opacity=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Line drawable for plotting segments and polylines.

    Arguments:
        vertices: `array_like`.
            Array with (x, y, z) coordinates of segment endpoints.
        indices: `array_like`.
            Array of vertex indices: int pair of indices from vertices array.
        indices_type: `str`.
            Interpretation of indices array
            Legal values are:

            :`segment`: indices contains pair of values,

            :`triangle`: indices contains triple of values
        color: `int`.
            Packed RGB color of the lines (0xff0000 is red, 0xff is blue) when `colors` is empty.
        colors: `array_like`.
            Array of int: packed RGB colors (0xff0000 is red, 0xff is blue) when attribute,
            color_map and color_range are empty.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        shader: `str`.
            Display style (name of the shader used) of the lines.
            Legal values are:

            :`simple`: simple lines,

            :`thick`: thick lines,

            :`mesh`: high precision triangle mesh of segments (high quality and GPU load).
        shininess: `float`.
            Shininess of object material.
        radial_segments: 'int'.
            Number of segmented faces around the circumference of the tube
        width: `float`.
            Thickness of the lines.
        opacity: `float`.
            Opacity of lines.
        name: `string`.
            A name of a object
        group: `string`.
            A name of a group
        custom_data: `dict`.
            A object with custom data attached to object.
            """
    if color_map is None:
        color_map = default_colormap

    color_map = (
        np.array(color_map, np.float32) if type(color_map) is not dict else color_map
    )
    attribute = (
        np.array(attribute, np.float32) if type(attribute) is not dict else attribute
    )
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Lines(
            vertices=vertices,
            indices=indices,
            indices_type=indices_type,
            color=color,
            width=width,
            shader=shader,
            shininess=shininess,
            radial_segments=radial_segments,
            colors=colors,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def line(
        vertices,
        color=_default_color,
        colors=[],  # lgtm [py/similar-function]
        attribute=[],
        color_map=None,
        color_range=[],
        width=0.01,
        opacity=1.0,
        shader="thick",
        shininess=50.0,
        radial_segments=8,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Line drawable for plotting segments and polylines.

    Parameters
    ----------
    vertices : array_like
        Array with (x, y, z) coordinates of segment endpoints.
    color : int, optional
        Hex color of the lines when `colors` is empty, by default _default_color.
    colors : list, optional
        Array of Hex colors when attribute, color_map and color_range are empty, by default [].
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    width : float, optional
        Thickness of the lines, by default 0.01.
    shader : {'simple', 'think', 'mesh'}, optional
        Display style of the lines, by default `thick`.
    shininess: `float`.
        Shininess of object material.
    radial_segments : int, optional
        Number of segmented faces around the circumference of the tube, by default 8.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    custom_data: `dict`
        A object with custom data attached to object.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    Line
        Line Drawable.
    """
    if color_map is None:
        color_map = default_colormap
    color_map = (
        np.array(color_map, np.float32) if type(
            color_map) is not dict else color_map
    )
    attribute = (
        np.array(attribute, np.float32) if type(
            attribute) is not dict else attribute
    )
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Line(
            vertices=vertices,
            color=color,
            width=width,
            shader=shader,
            radial_segments=radial_segments,
            colors=colors,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            shininess=shininess,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def mesh(
        vertices,
        indices,
        normals=[],
        color=_default_color,
        colors=[],
        attribute=[],
        color_map=None,
        # lgtm [py/similar-function]
        color_range=[],
        wireframe=False,
        flat_shading=True,
        shininess=50.0,
        opacity=1.0,
        texture=None,
        texture_file_format=None,
        volume=[],
        volume_bounds=[],
        opacity_function=[],
        side="front",
        uvs=None,
        slice_planes=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        triangles_attribute=[],
        **kwargs
):
    """Create a Mesh drawable from 3D triangles.

    Parameters
    ----------
    vertices : array_like
        Array of triangle vertices, `float` (x, y, z) coordinate triplets.
    indices : array_like
        Array of vertex indices. `int` triplets of indices from vertices array.
    normals: array_like, optional
        Array of vertex normals: float (x, y, z) coordinate triples. Normals are used when flat_shading is false.
        If the normals are not specified here, normals will be automatically computed.
    color : int, optional
        Hex color of the vertices when `colors` is empty, by default _default_color.
    colors : list, optional
        Array of Hex colors when attribute, color_map and color_range are empty, by default [].
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    wireframe : bool, optional
        Display the mesh as wireframe, by default False.
    flat_shading : bool, optional
        Display the mesh with flat shading, by default True.
    shininess: `float`.
        Shininess of object material.
    opacity : float, optional
        Opacity of the mesh, by default 1.0.
    texture : bytes, optional
        Image data in a specific format, by default None.
    texture_file_format : str, optional
        Format of the data, , by default None.
        It should be the second part of MIME format of type 'image/',e.g. 'jpeg', 'png', 'gif', 'tiff'.
    volume : list, optional
        3D array of `float`, by default [].
    volume_bounds : list, optional
        6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1), by default [].
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].
        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    side : {'front', 'back', 'double'}, optional
        Side to render, by default "front".
    uvs : array_like, optional
        float uvs for the texturing corresponding to each vertex, by default None.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    custom_data: `dict`
        A object with custom data attached to object.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    triangles_attribute : list, optional
        _description_, by default []
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    Mesh
        Mesh Drawable
    """
    if color_map is None:
        color_map = default_colormap
    color_map = (
        np.array(color_map, np.float32) if type(
            color_map) is not dict else color_map
    )
    uvs = np.array(uvs, np.float32) if type(uvs) is not dict else color_map
    attribute = (
        np.array(attribute, np.float32) if type(
            attribute) is not dict else attribute
    )
    normals = (
        np.array(normals, np.float32) if type(
            normals) is not dict else normals
    )
    triangles_attribute = (
        np.array(triangles_attribute, np.float32)
        if type(triangles_attribute) is not dict
        else triangles_attribute
    )
    volume_bounds = (
        np.array(volume_bounds, np.float32)
        if type(volume_bounds) is not dict
        else volume_bounds
    )

    if len(attribute) > 0:
        color_range = check_attribute_color_range(attribute, color_range)

    if len(triangles_attribute) > 0:
        color_range = check_attribute_color_range(
            triangles_attribute, color_range)

    if len(volume) > 0:
        color_range = check_attribute_color_range(volume, color_range)

    return process_transform_arguments(
        Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            color=color,
            colors=colors,
            attribute=attribute,
            triangles_attribute=triangles_attribute,
            color_map=color_map,
            color_range=color_range,
            wireframe=wireframe,
            flat_shading=flat_shading,
            shininess=shininess,
            opacity=opacity,
            volume=volume,
            volume_bounds=volume_bounds,
            opacity_function=opacity_function,
            side=side,
            texture=texture,
            uvs=uvs,
            texture_file_format=texture_file_format,
            slice_planes=slice_planes,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def stl(
        stl,
        color=_default_color,
        wireframe=False,
        flat_shading=True,
        shininess=50.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create an STL drawable for data in STereoLitograpy format.

    Parameters
    ----------
    stl : `str` or `bytes`
        STL data in either ASCII STL (`str`) or Binary STL (`bytes`).
    color : int, optional
        Hex color of the mesh, by default _default_color.
    wireframe : bool, optional
        Display the mesh as wireframe, by default False.
    flat_shading : bool, optional
        Display the mesh with flat shading, by default True.
    shininess: `float`.
        Shininess of object material.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    custom_data: `dict`
        A object with custom data attached to object.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    STL
        STL Drawable.
    """
    plain = isinstance(stl, six.string_types)

    return process_transform_arguments(
        STL(
            text=stl if plain else None,
            binary=stl if not plain else None,
            color=color,
            wireframe=wireframe,
            flat_shading=flat_shading,
            shininess=shininess,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def surface(
        heights,
        color=_default_color,
        wireframe=False,
        flat_shading=True,
        shininess=50.0,
        attribute=[],
        color_map=None,
        color_range=[],
        opacity=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Surface drawable.

    Plot a 2d function: z = f(x, y).
    The default domain of the scalar field is -0.5 < x, y < 0.5.

    If the domain should be different, the bounding box needs to be transformed using `kwargs`

    - ``surface(..., bounds=[-1, 1, -1, 1])``
    - ``surface(..., xmin=-10, xmax=10, ymin=-4, ymax=4)``

    Parameters
    ----------
    heights : array_like
        Array of `float` values.
    color : int, optional
        Hex color of the surface, by default _default_color.
    wireframe : bool, optional
        Display the mesh as wireframe, by default False.
    flat_shading : bool, optional
        Display the mesh with flat shading, by default True.
    shininess: `float`.
        Shininess of object material.
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
    opacity: `float`.
        Opacity of surface.
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    custom_data: `dict`
        A object with custom data attached to object.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    Surface
        Surface Drawable.
    """
    if color_map is None:
        color_map = default_colormap
    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Surface(
            heights=heights,
            color=color,
            wireframe=wireframe,
            flat_shading=flat_shading,
            shininess=shininess,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    ) 