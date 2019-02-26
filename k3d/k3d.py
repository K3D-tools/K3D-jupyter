# optional dependency
try:
    # noinspection PyPackageRequirements
    import vtk
    # noinspection PyPackageRequirements
    from vtk.util import numpy_support
except ImportError:
    vtk = None
    numpy_support = None

import numpy as np
import six
from .colormaps import basic_color_maps
from .plot import Plot
from .objects import (Line, MarchingCubes, Mesh, Points, STL, Surface, Text, Text2d, Texture, TextureText, VectorField,
                      Vectors, Volume, Voxels, SparseVoxels, VoxelsGroup, VoxelsIpyDW)
from .transform import process_transform_arguments
from .helpers import check_attribute_range

_default_color = 0x0000FF  # blue
nice_colors = (
    0xe6194b, 0x3cb44b, 0xffe119, 0x0082c8,
    0xf58231, 0x911eb4, 0x46f0f0, 0xf032e6,
    0xd2f53c, 0xfabebe, 0x008080, 0xe6beff,
    0xaa6e28, 0xfffac8, 0x800000, 0xaaffc3,
    0x808000, 0xffd8b1, 0x000080, 0x808080,
    0xFFFFFF, 0x000000
)


def line(vertices, color=_default_color, colors=[], attribute=[], color_map=[], color_range=[], width=0.01,
         shader='thick', radial_segments=8, compression_level=0, **kwargs):
    """Create a Line drawable for plotting segments and polylines.

    Arguments:
        vertices: `array_like`.
            Array with (x, y, z) coordinates of segment endpoints.
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
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""

    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_range(attribute, color_range)

    return process_transform_arguments(
        Line(vertices=vertices,
             color=color,
             width=width,
             shader=shader,
             radial_segments=radial_segments,
             colors=colors,
             attribute=attribute,
             color_map=color_map,
             color_range=color_range,
             compression_level=compression_level),
        **kwargs
    )


def marching_cubes(scalar_field, level, color=_default_color, wireframe=False, flat_shading=True, compression_level=0,
                   **kwargs):
    """Create a MarchingCubes drawable.

    Plot an isosurface of a scalar field obtained through a Marching Cubes algorithm.

    The default domain of the scalar field is -0.5 < x, y, z < 0.5.

    If the domain should be different, the bounding box needs to be transformed using kwargs, like this:
        marching_cubes(..., bounds=[-1, 1, -1, 1, -1, 1])
    or:
        marching_cubes(..., xmin=-10, xmax=10, ymin=-4, ymax=4, zmin=0, zmax=20)
    or:
        marching_cubes(..., scaling=[width, height, length])

    Arguments:
        scalar_field: `array_like`.
            A 3D scalar field of values.
        level: `float`.
            Value at the computed isosurface.
        color: `int`.
            Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        MarchingCubes(scalar_field=scalar_field,
                      color=color,
                      level=level,
                      wireframe=wireframe,
                      flat_shading=flat_shading,
                      compression_level=compression_level),
        **kwargs
    )


def mesh(vertices, indices, color=_default_color, attribute=[], color_map=[], color_range=[], wireframe=False,
         flat_shading=True, compression_level=0, **kwargs):
    """Create a Mesh drawable representing a 3D triangles mesh.

    Arguments:
        vertices: `array_like`.
            Array of triangle vertices: float (x, y, z) coordinate triplets.
        indices: `array_like`.
            Array of vertex indices: int triplets of indices from vertices array.
        color: `int`.
            Packed RGB color of the mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_range(attribute, color_range)

    return process_transform_arguments(
        Mesh(vertices=vertices,
             indices=indices,
             color=color,
             attribute=attribute,
             color_map=color_map,
             color_range=color_range,
             wireframe=wireframe,
             flat_shading=flat_shading,
             compression_level=compression_level),
        **kwargs
    )


def points(positions, colors=[], color=_default_color, point_size=1.0, shader='3dSpecular', opacity=1.0,
           compression_level=0, **kwargs):
    """Create a Points drawable representing a point cloud.

    Arguments:
        positions: `array_like`.
            Array with (x, y, z) coordinates of the points.
        colors: `array_like`.
            Same-length array of `int`-packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`.
            Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        opacity: `float`.
            Opacity of points.
        point_size: `float`.
            Diameter of the balls representing the points in 3D space.
        shader: `str`.
            Display style (name of the shader used) of the points.
            Legal values are:

            :`flat`: simple circles with uniform color,

            :`3d`: little 3D balls,

            :`3dSpecular`: little 3D balls with specular lightning,

            :`mesh`: high precision triangle mesh of a ball (high quality and GPU load).

        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Points(positions=positions, colors=colors,
               color=color, point_size=point_size, shader=shader,
               opacity=opacity,
               compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def stl(stl, color=_default_color, wireframe=False, flat_shading=True, compression_level=0, **kwargs):
    """Create an STL drawable for data in STereoLitograpy format.

    Arguments:
        stl: `str` or `bytes`.
            STL data in either ASCII STL (string) or Binary STL (bytes).
        color: `int`.
            Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    plain = isinstance(stl, six.string_types)

    return process_transform_arguments(
        STL(text=stl if plain else None,
            binary=stl if not plain else None,
            color=color,
            wireframe=wireframe,
            flat_shading=flat_shading,
            compression_level=compression_level),
        **kwargs
    )


def surface(heights, color=_default_color, wireframe=False, flat_shading=True, compression_level=0, **kwargs):
    """Create a Surface drawable.

    Plot a 2d function: z = f(x, y).

    The default domain of the scalar field is -0.5 < x, y < 0.5.

    If the domain should be different, the bounding box needs to be transformed using kwargs, like this:
        surface(..., bounds=[-1, 1, -1, 1])
    or:
        surface(..., xmin=-10, xmax=10, ymin=-4, ymax=4)

    Arguments:
        heights: `array_like`.
            A 2d scalar function values grid.
        color: `int`.
            Packed RGB color of the surface (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Surface(heights=heights, color=color, wireframe=wireframe, flat_shading=flat_shading,
                compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def text(text, position=(0, 0, 0), color=_default_color, reference_point='lb', size=1.0, compression_level=0):
    """Create a Text drawable for 3D-positioned text labels.

    Arguments:
        text: `str`.
            Content of the text.
        position: `list`.
            Coordinates (x, y, z) of the text's position.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`.
            Two-letter string representing the text's alignment.
            First letter: 'l', 'c' or 'r': left, center or right

            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`.
            Font size in 'em' HTML units."""
    return Text(position=position, reference_point=reference_point, text=text, size=size, color=color,
                compression_level=compression_level)


# noinspection PyShadowingNames
def text2d(text, position=(0, 0), color=_default_color, size=1.0, reference_point='lt', compression_level=0):
    """Create a Text2d drawable for 2D-positioned (viewport bound, OSD) labels.

    Arguments:
        text: `str`.
            Content of the text.
        position: `list`.
            Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`.
            Two-letter string representing the text's alignment.

            First letter: 'l', 'c' or 'r': left, center or right

            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`.
            Font size in 'em' HTML units."""
    return Text2d(position=position, reference_point=reference_point, text=text, size=size, color=color,
                  compression_level=compression_level)


def texture(binary=None, file_format=None, color_map=basic_color_maps.Rainbow, color_range=[], attribute=[],
            compression_level=0, **kwargs):
    """Create a Texture drawable for displaying 2D raster images in common formats.

    By default, the texture image is mapped into the square: -0.5 < x, y < 0.5, z = 1.
    If the size (scale, aspect ratio) or position should be different then the texture should be transformed
    using kwargs, for example:
        texture(..., xmin=0, xmax=640, ymin=0, ymax=480)
    or:
        texture(..., bounds=[0, 10, 0, 20])
    or:
        texture(..., scaling=[1.0, 0.75, 0])

    Arguments:
        binary: `bytes`.
            Image data in a specific format.
        file_format: `str`.
            Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        attribute: `array_like`.
            Array of float attribute for the color mapping, corresponding to each pixels.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""

    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_range(attribute, color_range)

    return process_transform_arguments(
        Texture(binary=binary,
                file_format=file_format,
                color_map=color_map,
                color_range=color_range,
                attribute=attribute,
                compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def texture_text(text, position=(0, 0, 0), color=_default_color, font_weight=400, font_face='Courier New',
                 font_size=68, size=1.0, compression_level=0):
    """Create a TextureText drawable.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Arguments:
        text: `str`.
            Content of the text.
        position: `list`.
            Coordinates (x, y, z) of the text's position.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        size: `float`.
            Size of the texture sprite containing the text.
        font_face: `str`.
            Name of the font to use for rendering the text.
        font_weight: `int`.
            Thickness of the characters in HTML-like units from the range (100, 900), where
            400 is normal and 600 is bold font.
        font_size: `int`.
            The font size inside the sprite texture in px units. This does not affect the size of the
            text in the scene, only the accuracy and raster size of the texture."""
    return TextureText(text=text, position=position, color=color, size=size,
                       font_face=font_face, font_size=font_size, font_weight=font_weight,
                       compression_level=compression_level)


# noinspection PyShadowingNames
def vector_field(vectors,
                 colors=[],
                 origin_color=None, head_color=None, color=_default_color,
                 use_head=True, head_size=1.0, scale=1.0, line_width=0.01, compression_level=0, **kwargs):
    """Create a VectorField drawable for displaying dense 2D or 3D grids of vectors of same dimensionality.

    By default, the origins of the vectors are assumed to be a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    or -0.5 < x, y < 0.5 square, regardless of the passed vector field shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using kwargs:
        vector_field(..., bounds=[-pi, pi, -pi, pi, 0, 1])
    or:
        vector_field(..., scaling=[scale_x, scale_y, scale_z]).

    For sparse (i.e. not forming a grid) 3D vectors, use the `vectors()` function.

    Arguments:
        vectors: `array_like`.
            Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
        colors: `array_like`.
            Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`.
            Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`.
            Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        color: `int`.
            Packed RGB color of the vectors (0xff0000 is red, 0xff is blue) when `colors` is empty and
            origin_color and head_color are not specified.
        use_head: `bool`.
            Whether vectors should display an arrow head.
        head_size: `float`.
            The size of the arrow heads.
        scale: `float`.
            Scale factor for the vector lengths, for artificially scaling the vectors in place.
        line_width: `float`.
            Width of the vector segments.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        VectorField(vectors=vectors,
                    colors=colors,
                    use_head=use_head,
                    head_size=head_size,
                    line_width=line_width,
                    head_color=head_color if head_color is not None else color,
                    origin_color=origin_color if origin_color is not None else color,
                    scale=scale,
                    compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def vectors(origins, vectors=None, colors=[],
            origin_color=None, head_color=None, color=_default_color,
            use_head=True, head_size=1.0,
            labels=[], label_size=1.0,
            line_width=0.01, compression_level=0, **kwargs):
    """Create a Vectors drawable representing individual 3D vectors.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For dense (i.e. forming a grid) 3D or 2D vectors, use the `vector_field` function.

    Arguments:
        origins: `array_like`.
            Array of (x, y, z) coordinates of vector origins, when `vectors` is None, these
            are (dx, dy, dz) components of unbound vectors (which are displayed as originating in (0, 0, 0)).
        vectors: `array_like`.
            The vectors as (dx, dy, dz) float triples. When not given, the `origins` are taken
            as vectors. When given, it must be same size as `origins`.
        colors: `array_like`.
            Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`.
            Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`.
            Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        color: `int`.
            Packed RGB color of the vectors (0xff0000 is red, 0xff is blue) when `colors` is empty and
            origin_color and head_color are not specified.
        use_head: `bool`.
            Whether vectors should display an arrow head.
        head_size: `float`.
            The size of the arrow heads.
        labels: `list` of `str`.
            Captions to display next to the vectors.
        label_size: `float`.
            Label font size in 'em' HTML units.
        line_width: `float`.
            Width of the vector segments.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
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
            compression_level=compression_level
        ),
        **kwargs
    )


# noinspection PyShadowingNames
def voxels(voxels, color_map=nice_colors, wireframe=False, outlines=True, outlines_color=0, opacity=1.0,
           compression_level=0,
           **kwargs):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        voxels(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        voxels(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        voxels: `array_like`.
            3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).

            The color defined at index i is for voxel value (i+1), e.g.:

            color_map = [0xff, 0x00ff]

            voxels =
            [[[
                0, # empty voxel

                1, # blue voxel

                2  # red voxel
            ]]]
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Voxels(voxels=voxels, color_map=color_map, wireframe=wireframe,
               outlines=outlines, outlines_color=outlines_color, opacity=opacity,
               compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def sparse_voxels(sparse_voxels, space_size, color_map=nice_colors, wireframe=False, outlines=True, outlines_color=0,
                  opacity=1.0, compression_level=0,
                  **kwargs):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        voxels(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        voxels(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        sparse_voxels: `array_like`.
            2D array of `coords` in format [[x,y,z,v],[x,y,z,v]].
            v = 0 means empty voxel, 1 and above refer to consecutive color_map entries.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        SparseVoxels(sparse_voxels=sparse_voxels, space_size=space_size, color_map=color_map, wireframe=wireframe,
                     outlines=outlines, outlines_color=outlines_color, opacity=opacity,
                     compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def voxels_group(voxels_group, space_size, color_map=nice_colors, wireframe=False, outlines=True, outlines_color=0,
                 opacity=1.0, compression_level=0, **kwargs):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        voxels(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        voxels(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        voxels_group: `array_like`.
            List of `chunks` in format {voxels: np.array, coord: [x,y,z], multiple: number}.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""

    for group in voxels_group:
        group['coord'] = np.array(group['coord'])
        group['voxels'] = np.array(group['voxels'])

        if 'multiple' not in group:
            group['multiple'] = 1

    return process_transform_arguments(
        VoxelsGroup(voxels_group=voxels_group, space_size=space_size, color_map=color_map, wireframe=wireframe,
                    outlines=outlines, outlines_color=outlines_color, opacity=opacity,
                    compression_level=compression_level),
        **kwargs
    )


# noinspection PyShadowingNames
def volume(volume, color_map, color_range=[], samples=512.0, alpha_coef=50.0, gradient_step=0.005, shadow='off',
           shadow_delay=500, shadow_res=128, compression_level=0, **kwargs):
    """Create a Volume drawable for 3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        volume(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        volume(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        volume: `array_like`.
            3D array of `float`
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).

            The color defined at index i is for voxel value (i+1), e.g.:

            color_map = [0xff, 0x00ff]

            voxels =
            [[[
                0, # empty voxel

                1, # blue voxel

                2  # red voxel
            ]]]
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of volume attribute mapped
            to 0 and 1 in the color map respectively.
        samples: `float`.
            Number of iteration per 1 unit of space.
        alpha_coef: `float`
            Alpha multiplier.
        gradient_step: `float`
            Gradient light step.
        shadow: `str`.
            Type of shadow on volume
            Legal values are:
                :`off`: shadow disabled,

                :`on_demand`: update shadow map on demand,

                :`dynamic`: update shadow map automaticaly every shadow_delay.
        shadow_delay: `float`.
            Minimum number of miliseconds between shadow map updates.
        shadow_res: `int`.
            Resolution of shadow map.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""

    color_range = check_attribute_range(volume, color_range)

    return process_transform_arguments(
        Volume(volume=volume, color_map=color_map, color_range=color_range, compression_level=compression_level,
               samples=samples, alpha_coef=alpha_coef, gradient_step=gradient_step, shadow=shadow,
               shadow_delay=shadow_delay, shadow_res=shadow_res),
        **kwargs)


def vtk_poly_data(poly_data, color=_default_color, color_attribute=None, color_map=basic_color_maps.Rainbow,
                  wireframe=False, compression_level=0, **kwargs):
    """Create a Mesh drawable from given vtkPolyData.

    This function requires the vtk module (from package VTK) to be installed.

    Arguments:
        poly_data: `vtkPolyData`.
            Native vtkPolyData geometry.
        color: `int`.
            Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        color_attribute: `tuple` of (`str`, `float`, `float`).
            This determines which scalar should be taken as the
            attribute for the color_map, and the color_range for the mesh: (attribute_name, min_value, max_value).
            A VTK mesh can have multiple named attributes in the vertices.
            min_value is the value mapped to 0 in the color_map.
            max_value is the value mapped to 1 in the color_map.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    if vtk is None:
        raise RuntimeError('vtk module is not available')

    if poly_data.GetPolys().GetMaxCellSize() > 3:
        cut_triangles = vtk.vtkTriangleFilter()
        cut_triangles.SetInputData(poly_data)
        cut_triangles.Update()
        poly_data = cut_triangles.GetOutput()

    if color_attribute is not None:
        attribute = numpy_support.vtk_to_numpy(poly_data.GetPointData().GetArray(color_attribute[0]))
        color_range = color_attribute[1:3]
    else:
        attribute = []
        color_range = []

    vertices = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    indices = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

    return process_transform_arguments(
        Mesh(vertices=np.array(vertices, np.float32),
             indices=np.array(indices, np.uint32),
             color=color,
             attribute=np.array(attribute, np.float32),
             color_range=color_range,
             color_map=np.array(color_map, np.float32),
             wireframe=wireframe,
             compression_level=compression_level),
        **kwargs
    )


def plot(height=512,
         antialias=True,
         background_color=0xffffff,
         camera_auto_fit=True,
         grid_auto_fit=True,
         menu_visibility=True,
         voxel_paint_color=0,
         grid=(-1, -1, -1, 1, 1, 1)):
    """Create a K3D Plot widget.

    This creates the main widget for displaying 3D objects.

    Arguments:
        height: `int`.
            Height of the widget in pixels.
        antialias: `bool`.
            Enable antialiasing in WebGL renderer.
        background_color: `int`.
            Packed RGB color of the plot background (0xff0000 is red, 0xff is blue).
        camera_auto_fit: `bool`.
            Enable automatic camera setting after adding, removing or changing a plot object.
        grid_auto_fit: `bool`.
            Enable automatic adjustment of the plot grid to contained objects.
        menu_visibility: `bool`.
            Enable menu on GUI.
        voxel_paint_color: `int`.
            The (initial) int value to be inserted when editing voxels.
        grid: `array_like`.
            6-element tuple specifying the bounds of the plot grid (x0, y0, z0, x1, y1, z1)."""
    return Plot(antialias=antialias,
                background_color=background_color,
                camera_auto_fit=camera_auto_fit, grid_auto_fit=grid_auto_fit,
                height=height, menu_visibility=menu_visibility,
                voxel_paint_color=voxel_paint_color, grid=grid)


# noinspection PyShadowingNames
def voxels_ipydw(voxels, color_map, wireframe=False, outlines=True, outlines_color=0, compression_level=0, **kwargs):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        voxels(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        voxels(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        voxels: `array_like`.
            3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).

            The color defined at index i is for voxel value (i+1), e.g.:

            color_map = [0xff, 0x00ff]

            voxels =
            [[[
                0, # empty voxel

                1, # blue voxel

                2  # red voxel
            ]]]
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
        kwargs: `dict`.
            Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        VoxelsIpyDW(voxels=voxels, color_map=color_map, wireframe=wireframe,
                    outlines=outlines, outlines_color=outlines_color,
                    compression_level=compression_level),
        **kwargs
    )
