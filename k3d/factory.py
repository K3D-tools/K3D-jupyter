# optional dependency
try:
    # noinspection PyPackageRequirements
    import vtk

    # noinspection PyPackageRequirements
    from vtk.util import numpy_support as nps
except ImportError:
    vtk = None
    nps = None

import numpy as np
import six

from .colormaps import matplotlib_color_maps
from .helpers import check_attribute_color_range
from .objects import (
    Line,
    Lines,
    MarchingCubes,
    Mesh,
    Points,
    STL,
    Surface,
    Text,
    Text2d,
    Texture,
    TextureText,
    VectorField,
    Vectors,
    Volume,
    MIP,
    Voxels,
    SparseVoxels,
    VoxelsGroup,
    VoxelChunk,
    Label,
)
from .plot import Plot
from .transform import process_transform_arguments

_default_color = 0x0000FF  # blue
nice_colors = (
    0xE6194B,
    0x3CB44B,
    0xFFE119,
    0x0082C8,
    0xF58231,
    0x911EB4,
    0x46F0F0,
    0xF032E6,
    0xD2F53C,
    0xFABEBE,
    0x008080,
    0xE6BEFF,
    0xAA6E28,
    0xFFFAC8,
    0x800000,
    0xAAFFC3,
    0x808000,
    0xFFD8B1,
    0x000080,
    0x808080,
    0xFFFFFF,
    0x000000,
)

default_colormap = matplotlib_color_maps.Inferno


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
        radial_segments: 'int'.
            Number of segmented faces around the circumference of the tube
        width: `float`.
            Thickness of the lines.
        opacity: `float`.
            Opacity of line.
        name: `string`.
            A name of a object
        group: `string`.
            A name of a group
        custom_data: `dict`
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
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def marching_cubes(
        scalar_field,
        level,
        color=_default_color,
        wireframe=False,
        flat_shading=True,
        opacity=1.0,
        spacings_x=[],
        spacings_y=[],
        spacings_z=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a MarchingCubes drawable.

    Plot an isosurface of a scalar field obtained through
    `Marching cubes <https://en.wikipedia.org/wiki/Marching_cubes>`_ algorithm.
    The default domain of the scalar field is -0.5 < x, y, z < 0.5.

    If the domain should be different, the bounding box needs to be transformed using `kwargs`

    - ``marching_cubes(..., bounds=[-1, 1, -1, 1, -1, 1])``
    - ``marching_cubes(..., xmin=-10, xmax=10, ymin=-4, ymax=4, zmin=0, zmax=20)``
    - ``marching_cubes(..., scaling=[width, height, length])``

    Parameters
    ----------
    scalar_field : array_like
        3D scalar field of values.
    level : float
        Value at the computed isosurface.
    color : int, optional
        Hex color of the isosurface, by default _default_color.
    wireframe : bool, optional
        Display the mesh as wireframe, by default False.
    flat_shading : bool, optional
        Display the mesh with flat shading, by default True.
    opacity : float, optional
        Opacity of the mesh, by default 1.0.
    spacings_x : list, optional
        Spacings in x axis, by default [].
        Should match `scalar_field` shape.
    spacings_y : list, optional
        Spacings in y axis, by default [].
        Should match `scalar_field` shape.
    spacings_z : list, optional
        Spacings in z axis, by default [].
        Should match `scalar_field` shape.
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
    MarchingCubes
        MarchingCubes Drawable.
    """
    return process_transform_arguments(
        MarchingCubes(
            scalar_field=scalar_field,
            spacings_x=spacings_x,
            spacings_y=spacings_y,
            spacings_z=spacings_z,
            color=color,
            level=level,
            wireframe=wireframe,
            flat_shading=flat_shading,
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
        opacity=1.0,
        texture=None,
        texture_file_format=None,
        volume=[],
        volume_bounds=[],
        opacity_function=[],
        side="front",
        uvs=None,
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
    side : {'front', 'back', 'both'}, optional
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
            opacity=opacity,
            volume=volume,
            volume_bounds=volume_bounds,
            opacity_function=opacity_function,
            side=side,
            texture=texture,
            uvs=uvs,
            texture_file_format=texture_file_format,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def points(
        positions,
        colors=[],
        color=_default_color,
        point_size=1.0,
        point_sizes=[],
        shader="3dSpecular",
        opacity=1.0,
        opacities=[],
        attribute=[],
        color_map=None,
        color_range=[],
        opacity_function=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        mesh_detail=2,
        **kwargs
):
    """Create a Points drawable representing a point cloud.

    Parameters
    ----------
    positions : array_like
        Array of (x, y, z) coordinates.
    colors : list, optional
        Array of Hex colors, by default [].
    color : int, optional
        Hex color of the points when `colors` is empty, by default _default_color.
    point_size : float, optional
        Diameter of the points, by default 1.0.
    point_sizes : list, optional
        Same-length array of `float` sizes of the points, by default [].
    shader : {'flat', 'dot', '3d', '3dSpecular', 'mesh'}, optional
        Display style of the points, by default "3dSpecular".
    opacity : float, optional
        Opacity of points, by default 1.0.
    opacities : list, optional
        Same-length array of `float` opacity of the points, by default [].
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].
        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    custom_data: `dict`
        A object with custom data attached to object.
    mesh_detail : int, optional
        Detail level of points mesh, by default 2.
        Only valid if `shader` is set to `mesh`. Setting this to a value greater than 0 adds more vertices making it no longer an
        icosahedron. When detail is greater than 1, it's effectively a sphere.
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    Points
        Points Drawable.
    """
    if color_map is None:
        color_map = default_colormap

    attribute = (
        np.array(attribute, np.float32) if type(
            attribute) is not dict else attribute
    )
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Points(
            positions=positions,
            colors=colors,
            color=color,
            point_size=point_size,
            point_sizes=point_sizes,
            shader=shader,
            opacity=opacity,
            opacities=opacities,
            mesh_detail=mesh_detail,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            opacity_function=opacity_function,
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
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
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


# noinspection PyShadowingNames
def text(
        text,
        position=[0, 0, 0],
        color=_default_color,
        reference_point="lb",
        on_top=True,
        size=1.0,
        label_box=True,
        is_html=False,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Text drawable for 3D-positioned text labels.

    Parameters
    ----------
    text: str or list of str
        Content of the text.
    position : list
        (x, y, z) coordinates of text position, by default (0, 0, 0).
        If n text is pass position should contain 3*n elements .
    color : int, optional
        Hex color of the text, by default _default_color.
    reference_point : str, optional
        Two-letter string representing text alignment, by default "lb".

        First letters

        - ``l`` -- left
        - ``c`` -- center
        - ``r`` -- right

        Second letters

        - ``t`` -- top
        - ``c`` -- center
        - ``b`` -- bottom
    on_top : bool, optional
        Render order with 3d object, by default True.
    size : float, optional
        Font size in 'em' HTML units, by default 1.0.
    label_box : bool, optional
        Label background box, by default True.
    is_html : bool, optional
        Interprete text as HTMl instead of KaTeX, by default False.
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
    Text
        Text Drawable.
    """
    return process_transform_arguments(
        Text(
            position=position,
            reference_point=reference_point,
            text=text,
            size=size,
            color=color,
            on_top=on_top,
            is_html=is_html,
            label_box=label_box,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def text2d(
        text,
        position=(0, 0),
        color=_default_color,
        size=1.0,
        reference_point="lt",
        label_box=True,
        is_html=False,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
):
    """Create a Text2d drawable for 2D-positioned (viewport bound, OSD) labels.

    Parameters
    ----------
    text : str or list of str
        Text content.
    position : tuple, optional
        (rx, ry) text position ratios in range (0, 1) - relative to canvas size, by default (0, 0).
        If n text is pass position should contain 2*n elements .
    color : int, optional
        Hex color of the text, by default _default_color.
    reference_point : str, optional
        Two-letter string representing text alignment, by default "lb".

        First letters

        - `l` -- left
        - `c` -- center
        - `r` -- right

        Second letters

        - `t` -- top
        - `c` -- center
        - `b` -- bottom
    size : float, optional
        Font size in 'em' HTML units, by default 1.0.
    label_box : bool, optional
        Label background box, by default True.
    is_html : bool, optional
        Interprete text as HTMl instead of KaTeX, by default False.
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
    Text2d
        Text2d Drawable.
    """
    return Text2d(
        position=position,
        reference_point=reference_point,
        text=text,
        size=size,
        color=color,
        is_html=is_html,
        label_box=label_box,
        name=name,
        group=group,
        custom_data=custom_data,
        compression_level=compression_level,
    )


# noinspection PyShadowingNames
def label(
        text,
        position=(0, 0, 0),
        color=_default_color,
        on_top=True,
        size=1.0,
        max_length=0.8,
        mode="dynamic",
        is_html=False,
        label_box=True,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Text drawable for 3D-positioned text labels.

    Parameters
    ----------
    text: str or list of str
        Content of the text.
    position : list
        (x, y, z) coordinates of text position, by default (0, 0, 0).
        If n text is pass position should contain 3*n elements .
    color : int, optional
        Hex color of the text, by default _default_color.
    on_top : bool, optional
        Render order with 3d object, by default True.
    size : float, optional
        Font size in 'em' HTML units, by default 1.0.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    max_length : float, optional
        Maximum length of line in % of half screen size (only when `mode` is `dynamic`), by default 0.8.
    mode : {'dynamic', 'local', 'side'}, optional
        Label node, by default "dynamic".
    is_html : bool, optional
        Interprete text as HTMl instead of KaTeX, by default False.
    label_box : bool, optional
        Label background box, by default True.
    custom_data: `dict`
        A object with custom data attached to object.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.
    **kwargs
        For other keyword-only arguments, see :ref:`process_transform_arguments`.

    Returns
    -------
    Label
        Label Drawable.
    """
    return process_transform_arguments(
        Label(
            position=position,
            text=text,
            size=size,
            color=color,
            on_top=on_top,
            max_length=max_length,
            mode=mode,
            is_html=is_html,
            label_box=label_box,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def texture(
        binary=None,
        file_format=None,
        color_map=None,
        color_range=[],
        attribute=[],
        puv=[],
        opacity_function=[],
        interpolation=True,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Texture drawable for displaying 2D raster images in common formats.

    By default, the texture image is mapped into the square: -0.5 < x, y < 0.5, z = 1.

    If the size (scale, aspect ratio) or position should be different then the texture should be transformed
    using `kwargs`

    - ``texture(..., xmin=0, xmax=640, ymin=0, ymax=480)``
    - ``texture(..., bounds=[0, 10, 0, 20])``
    - ``texture(..., scaling=[1.0, 0.75, 0])``

    Parameters
    ----------
    binary : bytes, optional
        Image data in a specific format, by default None
    file_format : str, optional
        Format of the data, by default None.

        It should be the second part of MIME format of type 'image/',e.g. 'jpeg', 'png', 'gif', 'tiff'.
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    attribute: list, optional
        List of values used to apply `color_map`, by default [].
    puv : list, optional
        List of `float` triplets (x,y,z), by default [].
        The first triplet mean a position of left-bottom corner of texture.
        Second and third triplets means a base of coordinate system for texture.
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].
        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    interpolation : bool, optional
        Interpolate the data, by default True
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
    Texture
        Texture Drawable.
    """
    if color_map is None:
        color_map = default_colormap
    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Texture(
            binary=binary,
            file_format=file_format,
            color_map=color_map,
            color_range=color_range,
            attribute=attribute,
            opacity_function=opacity_function,
            puv=puv,
            interpolation=interpolation,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def texture_text(
        text,
        position=(0, 0, 0),
        color=_default_color,
        font_weight=400,
        font_face="Courier New",
        font_size=68,
        size=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a TextureText drawable.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Parameters
    ----------
    text: str or list of str
        Content of the text.
    position : list
        (x, y, z) coordinates of text position, by default (0, 0, 0).
        If n text is pass position should contain 3*n elements .
    color : int, optional
        Hex color of the text, by default _default_color.
    font_weight : int, optional
        Characters thickness in HTML-like units [100, 900], by default 400.
    font_face : str, optional
        Font name used to render text, by default "Courier New".
    font_size : int, optional
        Font size inside the sprite texture in px units, by default 68.
        This does not affect the size of the text in the scene,
        only the accuracy and raster size of the texture.
    size : float, optional
        Size of the texture sprite containing the text, by default 1.0.
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
    TextureText
        TextureText Drawable.
    """
    return process_transform_arguments(
        TextureText(
            text=text,
            position=position,
            color=color,
            size=size,
            font_face=font_face,
            font_size=font_size,
            font_weight=font_weight,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def vector_field(
        vectors,
        colors=[],
        origin_color=None,
        head_color=None,
        color=_default_color,
        use_head=True,
        head_size=1.0,
        scale=1.0,
        line_width=0.01,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a VectorField drawable for displaying dense 2D or 3D grids of vectors of same dimensionality.

    By default, the origins of the vectors are assumed to be a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    or -0.5 < x, y < 0.5 square, regardless of the passed vector field shape, like aspect ratio.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``vector_field(..., bounds=[-pi, pi, -pi, pi, 0, 1])``
    - ``vector_field(..., scaling=[scale_x, scale_y, scale_z])``

    For sparse (i.e. not forming a grid) 3D vectors, use `vectors`.

    Parameters
    ----------
    vectors : array_like
        Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
    colors : list, optional
        Array of Hex colors of vectors, by default [].

        The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
    origin_color : int, optional
        Hex color of vector origins when `colors` is empty, by default None.
    head_color : int, optional
        Hex color of vector heads when `colors` is empty, by default None.
    color : int, optional
        Hex color of the vectors when `colors` is empty, by default _default_color.
    use_head : bool, optional
        Display vector heads, by default True.
    head_size : float, optional
        Vector heads size, by default 1.0.
    scale : float, optional
        Scale factor for the vector lengths, by default 1.0.
    line_width : float, optional
        Width of vector segments, by default 0.01.
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
    VectorField
        VectorField Drawable.
    """
    return process_transform_arguments(
        VectorField(
            vectors=vectors,
            colors=colors,
            use_head=use_head,
            head_size=head_size,
            line_width=line_width,
            head_color=head_color if head_color is not None else color,
            origin_color=origin_color if origin_color is not None else color,
            scale=scale,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def vectors(
        origins,
        vectors=None,
        colors=[],
        origin_color=None,
        head_color=None,
        color=_default_color,
        use_head=True,
        head_size=1.0,
        labels=[],
        label_size=1.0,
        line_width=0.01,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Vectors drawable representing individual 3D vectors.

    For dense 3D or 2D vectors,like forming a grid, use `vector_field`.

    Parameters
    ----------
    origins : array_like
        Array of (x, y, z) coordinates of vector origins.
    vectors : array_like, optional
        Array of (dx, dy, dz) directions of vectors, by default None.
        Must have the same size as `origins`.
    colors : list, optional
        Array of Hex colors of vectors, by default [].
    origin_color : int, optional
        Hex color of vector origins when `colors` is empty, by default None.
    head_color : int, optional
        Hex color of vector heads when `colors` is empty, by default None.
    color : int, optional
        Hex color of the vectors when `colors` is empty, by default None.
    use_head : bool, optional
        Display vector heads, by default True.
    head_size : float, optional
        Vector heads size, by default 1.0.
    labels : list, optional
        List of `str` of caption the display next tot the vectors, by default [].
    label_size : float, optional
        Label font size in 'em' HTML units, by default 1.0.
    line_width : float, optional
        Width of vector segments, by default 0.01.
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
    Vectors
        Vectors Drawable.
    """
    return process_transform_arguments(
        Vectors(
            vectors=vectors if vectors is not None else origins,
            origins=origins if vectors is not None else np.zeros_like(vectors),
            colors=colors,
            origin_color=origin_color if origin_color is not None else color,
            head_color=head_color if head_color is not None else color,
            use_head=use_head,
            head_size=head_size,
            labels=labels,
            label_size=label_size,
            line_width=line_width,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def voxels(
        voxels,
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        bounds=None,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape, like aspect ratio.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``voxels(..., bounds=[0, 300, 0, 400, 0, 500])``
    - ``voxels(..., scaling=[scale_x, scale_y, scale_z])``

    Parameters
    ----------
    voxels : array_like
        3D array of `int` from 0 to 255.
        0 means empty voxel; 1 and above refer to consecutive `color_map`.
    color_map : int, optional
        List of Hex color, by default None.
        The color defined at index i is for voxel value (i+1).
    wireframe : bool, optional
        Display voxels as wireframe, by default False.
    outlines : bool, optional
        Display voxels outlines, by default True.
    outlines_color : int, optional
        Hex color of voxels outlines, by default 0.
    opacity : float, optional
        Opacity of voxels, by default 1.0.
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
    Voxels
        Voxels Drawable.
    """
    if color_map is None:
        color_map = nice_colors

    if bounds is not None:
        kwargs["bounds"] = bounds
    else:
        max_z, max_y, max_x = np.shape(voxels)
        kwargs["bounds"] = np.array([0, max_x, 0, max_y, 0, max_z])

    return process_transform_arguments(
        Voxels(
            voxels=voxels,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def sparse_voxels(
        sparse_voxels,
        space_size,
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        bounds=None,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a SparseVoxels drawable for 3D volumetric data.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``sparse_voxels(..., bounds=[0, 300, 0, 400, 0, 500])``
    - ``sparse_voxels(..., scaling=[scale_x, scale_y, scale_z])``

    Parameters
    ----------
    sparse_voxels : array_like
        2D array of cordinates [x, y, z, v],  x, y, z >= 0 and 0<= v <= 255.
        v = 0 means empty voxel; v >= 1 refer to consecutive `color_map`.
    space_size : array_like
        Width, Height and Length of space.
    color_map : int, optional
        List of Hex color, by default None.
        The color defined at index i is for voxel value (i+1).
    wireframe : bool, optional
        Display voxels as wireframe, by default False.
    outlines : bool, optional
        Display voxels outlines, by default True.
    outlines_color : int, optional
        Hex color of voxels outlines, by default 0.
    opacity : float, optional
        Opacity of voxels, by default 1.0.
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
    SparseVoxels
        SparseVoxels Drawable.
    """
    if color_map is None:
        color_map = nice_colors

    assert (
            isinstance(space_size, (tuple, list, np.ndarray))
            and np.shape(space_size) == (3,)
            and all(d > 0 for d in space_size)
    )

    return process_transform_arguments(
        SparseVoxels(
            sparse_voxels=sparse_voxels,
            space_size=space_size,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def voxels_group(
        space_size,
        voxels_group=[],
        chunks_ids=[],
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a VoxelsGroup drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape, like aspect ratio.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``voxels_group(..., bounds=[0, 300, 0, 400, 0, 500])``
    - ``voxels_group(..., scaling=[scale_x, scale_y, scale_z])``

    Parameters
    ----------
    space_size : array_like
        Width, Height, Length of space. Must be non-negative.
    voxels_group : list, optional
        List of `voxel_chunk` in format {voxels: np.array, coord: [x,y,z], multiple: number}, by default [].
    chunks_ids : list, optional
        List of `voxels_chunk` id, by default [].
    color_map : int, optional
        List of Hex color, by default None.
        The color defined at index i is for voxel value (i+1).
    wireframe : bool, optional
        Display voxels as wireframe, by default False.
    outlines : bool, optional
        Display voxels outlines, by default True.
    outlines_color : int, optional
        Hex color of voxels outlines, by default 0.
    opacity : float, optional
        Opacity of voxels, by default 1.0.
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
    VoxelsGroup
        VoxelsGroup Drawable.
    """
    if color_map is None:
        color_map = nice_colors

    for g in voxels_group:
        g["coord"] = np.array(g["coord"])
        g["voxels"] = np.array(g["voxels"])

        if "multiple" not in g:
            g["multiple"] = 1

    return process_transform_arguments(
        VoxelsGroup(
            voxels_group=voxels_group,
            chunks_ids=chunks_ids,
            space_size=space_size,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def volume(
        volume,
        color_map=None,
        opacity_function=None,
        color_range=[],
        samples=512.0,
        alpha_coef=50.0,
        gradient_step=0.005,
        shadow="off",
        interpolation=True,
        shadow_delay=500,
        shadow_res=128,
        focal_length=0.0,
        focal_plane=100.0,
        ray_samples_count=16,
        mask=[],
        mask_opacities=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a Volume drawable for 3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape ,like aspect ratio.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``volume(..., bounds=[0, 300, 0, 400, 0, 500])``
    - ``volume(..., scaling=[scale_x, scale_y, scale_z])``

    Parameters
    ----------
    volume : ndarray
        3D array of `float`.
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].
        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    samples : float, optional
        Number of iteration per 1 unit of space, by default 512.0.
    alpha_coef : float, optional
        Alpha multiplier, by default 50.0.
    gradient_step : float, optional
        Gradient light step, by default 0.005.
    shadow : {'off', 'on_demand', 'dynamic'}, optional
        Type of shadow on volume, by default "off".
    interpolation : bool, optional
        Interpolate volume raycasting data, by default True.
    shadow_delay : int, optional
        Minimum number of miliseconds between shadow map updates, by default 500.
    shadow_res : int, optional
        Resolution of shadow map, by default 128.
    focal_length : float, optional
        Focal length of depth of field renderer, by default 0.0.
    focal_plane : float, optional
        Focal plane of depth of field renderer, by default 100.0.
    ray_samples_count : int, optional
        Number of rays for depth of field rendering, by default 16.
    mask: `array_like`.
        3D array of `int` in range (0, 255).
    mask_opacities: `array_like`.
        List of opacity values for mask.
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
    Volume
        Volume Drawable.
    """
    if color_map is None:
        color_map = default_colormap

    color_range = (
        check_attribute_color_range(volume, color_range)
        if type(color_range) is not dict
        else color_range
    )

    if opacity_function is None:
        opacity_function = [np.min(color_map[::4]),
                            0.0, np.max(color_map[::4]), 1.0]

    return process_transform_arguments(
        Volume(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            compression_level=compression_level,
            samples=samples,
            alpha_coef=alpha_coef,
            gradient_step=gradient_step,
            interpolation=interpolation,
            shadow=shadow,
            shadow_delay=shadow_delay,
            shadow_res=shadow_res,
            focal_plane=focal_plane,
            focal_length=focal_length,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
            ray_samples_count=ray_samples_count,
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def mip(
        volume,
        color_map=None,
        opacity_function=None,
        color_range=[],
        samples=512.0,
        gradient_step=0.005,
        interpolation=True,
        mask=[],
        mask_opacities=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
    """Create a MIP drawable for 3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape, like aspect ratio.

    Different grid size, shape and rotation can be obtained using `kwargs`

    - ``mip(..., bounds=[0, 300, 0, 400, 0, 500])``
    - ``mip(..., scaling=[scale_x, scale_y, scale_z])``

    Parameters
    ----------
    volume : ndarray
        3D array of `float`.
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].
        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    samples : float, optional
        Number of iteration per 1 unit of space, by default 512.0.
    gradient_step : float, optional
        Gradient light step, by default 0.005.
    interpolation : bool, optional
        Interpolate volume raycasting data, by default True.
    mask: `array_like`.
        3D array of `int` in range (0, 255).
    mask_opacities: `array_like`.
        List of opacity values for mask.
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
    MIP
        MIP Drawable.
    """
    if color_map is None:
        color_map = default_colormap

    color_range = (
        check_attribute_color_range(volume, color_range)
        if type(color_range) is not dict
        else color_range
    )

    if opacity_function is None:
        opacity_function = [np.min(color_map[::4]),
                            0.0, np.max(color_map[::4]), 1.0]

    return process_transform_arguments(
        MIP(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            samples=samples,
            gradient_step=gradient_step,
            interpolation=interpolation,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def vtk_poly_data(
        poly_data,
        color=_default_color,
        color_attribute=None,
        color_map=None,
        side="front",
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
    """Create a Mesh drawable from given vtkPolyData.

    Require the vtk module (from package VTK) to be installed.

    Parameters
    ----------
    poly_data : vtkPolyData
        Native vtkPolyData geometry.
    color : int, optional
        Hex color of the mesh when when not using `color_map`, by default _default_color.
    color_attribute : tuple, optional
        (`str`, `float`, `float`) to determine which scalar should be used
        for the `color_map` and the `color_range` (attribute_name, min_value, max_value), by default None.

        A VTK mesh can have multiple named attributes in the vertices

        - min_value is the value mapped to 0 in the color_map
        - max_value is the value mapped to 1 in the color_map
    cell_color_attribute : tuple, optional
        (`str`, `float`, `float`) to determine which scalar should be used
        for the `color_map` and the `color_range` (attribute_name, min_value, max_value), by default None.

        A VTK mesh can have multiple named attributes in the vertices

        - min_value is the value mapped to 0 in the color_map
        - max_value is the value mapped to 1 in the color_map
    color_map : list, optional
        List of `float` quadruplets (attribute value, R, G, B) sorted by attribute value, by default None.
        The first quadruplet should have value 0.0, the last 1.0;
        R, G, B are RGB color components in the range 0.0 to 1.0.
    color_range : list, optional
        [min_value, max_value] pair determining the levels of color attribute mapped
        to 0 and 1 in the colormap, by default [].
    side : {"front", "back", "both"}, optional
        Side to render, by default "front".
    wireframe : bool, optional
        Display the mesh as wireframe, by default False.
    opacity : float, optional
        Opacity of mesh, by default 1.0.
    opacity_function : list, optional
        `float` tuples (attribute value, opacity) sorted by attribute value, by default [].

        The first tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
    volume : list, optional
        3D array of `float`, by default [].
    volume_bounds : list, optional
        6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1), by default [].
    flat_shading : bool, optional
        Display the mesh with flat shading, by default True.
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
    Mesh
        Mesh Drawable.

    Raises
    ------
    RuntimeError
        vtk module is not available.
    """
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
    elif volume != []:
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
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def voxel_chunk(voxels, coord, multiple=1, compression_level=0):
    """Create a VoxelChunk that can be used for `voxels_group`.

    Parameters
    ----------
    voxels : array_like
        3D array of `int` from 0 to 255.
        0 means empty voxel; 1 and above refer to value of a colormap.
    coord : array_like
        Coordinates of the chunk.
    multiple : int, optional
        For future usage, by default 1.
    compression_level : int, optional
        Level of data compression [-1, 9], by default 0.

    Returns
    -------
    VoxelChunk
        Voxel chunk.
    """
    return VoxelChunk(
        voxels=np.array(voxels, np.uint8),
        coord=np.array(coord, np.uint32),
        multiple=multiple,
        compression_level=compression_level,
    )


def plot(
        height=512,
        antialias=3,
        logarithmic_depth_buffer=True,
        background_color=0xffffff,
        camera_auto_fit=True,
        grid_auto_fit=True,
        grid_visible=True,
        screenshot_scale=2.0,
        grid=(-1, -1, -1, 1, 1, 1),
        grid_color=0xe6e6e6,
        label_color=0x444444,
        lighting=1.5,
        menu_visibility=True,
        voxel_paint_color=0,
        colorbar_object_id=-1,
        camera_fov=60.0,
        time=0.0,
        axes=['x', 'y', 'z'],
        axes_helper=1.0,
        axes_helper_colors=[0xff0000, 0x00ff00, 0x0000ff],
        camera_mode="trackball",
        snapshot_type='full',
        auto_rendering=True,
        camera_no_zoom=False,
        camera_no_rotate=False,
        camera_no_pan=False,
        camera_rotate_speed=1.0,
        camera_zoom_speed=1.2,
        camera_pan_speed=0.3,
        camera_damping_factor=0.0,
        fps=25.0,
        minimum_fps=-1,
        fps_meter=False,
        name=None,
        custom_data=None
):
    """Create a Plot widget.

    Parameters
    ----------
    height : int, optional
        Height of the widget in pixels, by default 512.
    antialias : int, optional
        WebGL renderer antialiasing, by default 3.
    logarithmic_depth_buffer : bool, optional
        WebGL renderer logarithmic depth buffer, by default True.
    background_color : int, optional
        Hex color of plot background, by default 0xffffff.
    camera_auto_fit : bool, optional
        Automatic camera setting after adding, removing or modifying objects, by default True.
    grid_auto_fit : bool, optional
        Automatic grid adjustment to contained objects, by default True.
    grid_visible : bool, optional
        Display grid, by default True.
    screenshot_scale : float, optional
        Screenshot resolution multiplier, by default 2.0.
    grid : tuple, optional
        6-element tuple specifying grid bounds (x0, y0, z0, x1, y1, z1), by default (-1, -1, -1, 1, 1, 1).
    grid_color : int, optional
        Hex color of the grid, by default 0xe6e6e6.
    label_color : int, optional
        Hex color of labels, by default 0x444444.
    lighting : float, optional
        Lighting factor, by default 1.5.
    menu_visibility : bool, optional
        Display K3D panel, by default True.
    voxel_paint_color : int, optional
        (initial) `int` value to be inserted when editing voxels, by default 0.
    colorbar_object_id : int, optional
        Id of colorbar object, by default -1.
    camera_fov : float, optional
        Camera field of view, by default 60.0.
    time : float, optional
        Time value, by default 0.0.
    axes : list, optional
        Axes labels, by default ['x', 'y', 'z'].
    axes_helper : float, optional
        Axes helper size, by default 1.0.
    name : str, optional
        Object name, by default None.
    group : str, optional
        Name of a group, by default None.
    camera_mode : {'trackball', 'orbit', 'fly'}, optional
        Mode of camera, by default 'trackball'.
    snapshot_type : {'full', 'online', 'inline'}, optional
        Type of snapshot, by default 'full'.
    auto_rendering : bool, optional
        Auto rendering state, by default True.
    camera_no_zoom : bool, optional
        Lock camera zoom, by default False.
    camera_no_rotate : bool, optional
        Lock camera rotation, by default False.
    camera_no_pan : bool, optional
        Lock camera pan, by default False.
    camera_rotate_speed : float, optional
        Camera rotation speed, by default 1.0.
    camera_zoom_speed : float, optional
        Camera zoom speed, by default 1.2.
    camera_pan_speed : float, optional
        Camera pan speed, by default 0.3.
    camera_damping_factor : float, optional
        Camera intensity of damping, by default 0.0.
    fps : float, optional
        Animations FPS, by default 25.0.
    minimum_fps: `Float`.
            If negative then disabled. Set target FPS to adaptative resolution.
    custom_data: `dict`
        A object with custom data attached to object.

    Returns
    -------
    Plot
        Plot Widget.
    """
    return Plot(
        antialias=antialias,
        logarithmic_depth_buffer=logarithmic_depth_buffer,
        background_color=background_color,
        lighting=lighting,
        time=time,
        colorbar_object_id=colorbar_object_id,
        camera_auto_fit=camera_auto_fit,
        grid_auto_fit=grid_auto_fit,
        grid_visible=grid_visible,
        grid_color=grid_color,
        label_color=label_color,
        height=height,
        menu_visibility=menu_visibility,
        voxel_paint_color=voxel_paint_color,
        grid=grid,
        axes=axes,
        axes_helper=axes_helper,
        axes_helper_colors=axes_helper_colors,
        screenshot_scale=screenshot_scale,
        camera_fov=camera_fov,
        name=name,
        camera_mode=camera_mode,
        snapshot_type=snapshot_type,
        camera_no_zoom=camera_no_zoom,
        camera_no_rotate=camera_no_rotate,
        camera_no_pan=camera_no_pan,
        camera_rotate_speed=camera_rotate_speed,
        camera_zoom_speed=camera_zoom_speed,
        camera_damping_factor=camera_damping_factor,
        camera_pan_speed=camera_pan_speed,
        auto_rendering=auto_rendering,
        fps=fps,
        minimum_fps=minimum_fps,
        fps_meter=fps_meter,
        custom_data=custom_data
    )
