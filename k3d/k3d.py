# optional dependency

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    vtk = None
    numpy_support = None

import numpy as np
import six
from .colormaps.basic_color_maps import basic_color_maps
from .plot import Plot
from .objects import (Line, MarchingCubes, Mesh, Points, STL, Surface, Text, Text2d, Texture, TextureText, VectorField,
                      Vectors, Voxels)
from .transform import process_transform_arguments

_default_color = 0x0000FF  # blue


def line(vertices, color=_default_color, width=1, **kwargs):
    """Create a Line drawable for plotting segments and polylines.

    Arguments:
        vertices: `array_like`. Array with (x, y, z) coordinates of segment endpoints.
        color: `int`. Packed RGB color of the lines (0xff0000 is red, 0xff is blue).
        width: `float`. Thickness of the lines.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Line(vertices=np.array(vertices, np.float32), color=color, width=width),
        **kwargs
    )


def marching_cubes(scalar_field, level, color=_default_color, **kwargs):
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
        scalar_field: `array_like`. A 3D scalar field of values.
        level: `float`. Value at the computed isosurface.
        color: `int`. Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue).
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        MarchingCubes(scalar_field=np.array(scalar_field, np.float32),
                      color=color,
                      level=level),
        **kwargs
    )


def mesh(vertices, indices, color=_default_color, attribute=[], color_map=[], color_range=[], **kwargs):
    """Create a Mesh drawable representing a 3D triangles mesh.

    Arguments:
        vertices: `array_like`. Array of triangle vertices: float (x, y, z) coordinate triplets.
        indices: `array_like`.  Array of vertex indices: int triplets of indices from vertices array.
        color: `int`. Packed RGB color of the mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        attribute: `array_like`. Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`. A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`. A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Mesh(vertices=np.array(vertices, np.float32),
             indices=np.array(indices, np.uint32),
             color=color,
             attribute=np.array(attribute, np.float32),
             color_map=np.array(color_map, np.float32),
             color_range=color_range),
        **kwargs
    )


def points(positions, colors=[], color=_default_color, model_matrix=np.identity(4), point_size=1.0,
           shader='3dSpecular', **kwargs):
    """Create a Points drawable representing a point cloud.

    Arguments:
        positions: `array_like`. Array with (x, y, z) coordinates of the points.
        colors: `array_like`. Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`. Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        point_size: `float`. Diameter of the balls representing the points in 3D space.
        shader: `str`. Display style (name of the shader used) of the points.
            Legal values are:
            `flat`: simple circles with uniform color,
            `3d`: little 3D balls,
            `3dSpecular`: little 3D balls with specular lightning,
            `mesh`: high precision triangle mesh of a ball (high quality and GPU load).
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Points(positions=np.array(positions, np.float32), colors=np.array(colors, np.float32),
               color=color, point_size=point_size, shader=shader),
        **kwargs
    )


def stl(stl, color=_default_color, **kwargs):
    """Create an STL drawable for data in STereoLitograpy format.

    Arguments:
        stl: `str` or `bytes`. STL data in either ASCII STL (string) or Binary STL (bytes).
        color: `int`. Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    plain = isinstance(stl, six.string_types)
    return process_transform_arguments(
        STL(text=stl if plain else None,
            binary=[] if plain else np.fromstring(stl, dtype=np.uint8),  # allow_null doesn't really work for Array...
            color=color),
        **kwargs
    )


def surface(heights, color=_default_color, **kwargs):
    """Create a Surface drawable.

    Plot a 2d function: z = f(x, y).

    The default domain of the scalar field is -0.5 < x, y < 0.5.

    If the domain should be different, the bounding box needs to be transformed using kwargs, like this:
        surface(..., bounds=[-1, 1, -1, 1])
    or:
        surface(..., xmin=-10, xmax=10, ymin=-4, ymax=4)

    Arguments:
        heights: `array_like`. A 2d scalar function values grid.
        color: `int`. Packed RGB color of the surface (0xff0000 is red, 0xff is blue).
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Surface(heights=np.array(heights, np.float32), color=color),
        **kwargs
    )


def text(text, position=(0, 0, 0), color=_default_color, reference_point='lb', size=1.0):
    """Create a Text drawable for 3D-positioned text labels.

    Arguments:
        text: `str`. Content of the text.
        position: `list`. Coordinates (x, y, z) of the text's position.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`. Two-letter string representing the text's alignment.
            First letter: 'l', 'c' or 'r': left, center or right
            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`. Font size in 'em' HTML units."""
    return Text(position=position, reference_point=reference_point, text=text, size=size, color=color)


def text2d(text, position=(0, 0), color=_default_color, size=1.0, reference_point='lt'):
    """Create a Text2d drawable for 2D-positioned (viewport bound, OSD) labels.

    Arguments:
        text: `str`. Content of the text.
        position: `list`. Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`. Two-letter string representing the text's alignment.
            First letter: 'l', 'c' or 'r': left, center or right
            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`. Font size in 'em' HTML units."""
    return Text2d(position=position, reference_point=reference_point, text=text, size=size, color=color)


def texture(binary, file_format, **kwargs):
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
        binary: `bytes`. Image data in a specific format.
        file_format: `str`. Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Texture(binary=binary,
                file_format=file_format),
        **kwargs
    )


def texture_text(text, position=(0, 0, 0), color=_default_color, font_weight=400, font_face='Courier New',
                 font_size=68, size=1.0):
    """Create a TextureText drawable.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Arguments:
        text: `str`. Content of the text.
        position: `list`. Coordinates (x, y, z) of the text's position.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        size: `float`. Size of the texture sprite containing the text.
        font_face: `str`. Name of the font to use for rendering the text.
        font_weight: `int`. Thickness of the characters in HTML-like units from the range (100, 900), where
            400 is normal and 600 is bold font.
        font_size: `int`. The font size inside the sprite texture in px units. This does not affect the size of the
            text in the scene, only the accuracy and raster size of the texture."""
    return TextureText(text=text, position=position, color=color, size=size,
                       font_face=font_face, font_size=font_size, font_weight=font_weight)


def vector_field(vectors,
                 colors=[],
                 origin_color=None, head_color=None, color=_default_color,
                 use_head=True, head_size=1.0, scale=1.0, **kwargs):
    """Create a VectorField drawable for displaying dense 2D or 3D grids of vectors of same dimensionality.

    By default, the origins of the vectors are assumed to be a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    or -0.5 < x, y < 0.5 square, regardless of the passed vector field shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using kwargs:
        vector_field(..., bounds=[-pi, pi, -pi, pi, 0, 1])
    or:
        vector_field(..., scaling=[scale_x, scale_y, scale_z]).

    For sparse (i.e. not forming a grid) 3D vectors, use the `vectors()` function.

    Arguments:
        vectors: `array_like`. Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
        colors: `array_like`. Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`. Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`. Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        color: `int`. Packed RGB color of the vectors (0xff0000 is red, 0xff is blue) when `colors` is empty and
            origin_color and head_color are not specified.
        use_head: `bool`. Whether vectors should display an arrow head.
        head_size: `float`. The size of the arrow heads.
        scale: `float`. Scale factor for the vector lengths, for artificially scaling the vectors in place.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        VectorField(vectors=vectors,
                    colors=np.array(colors, np.uint32),
                    use_head=use_head,
                    head_size=head_size,
                    head_color=head_color if head_color is not None else color,
                    origin_color=origin_color if origin_color is not None else color,
                    scale=scale),
        **kwargs
    )


def vectors(vectors, origins, colors=[],
            origin_color=None, head_color=None, color=_default_color,
            use_head=True, head_size=1.0,
            labels=[], label_size=1.0,
            line_width=1, **kwargs):
    """Create a Vectors drawable representing individual 3D vectors.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For dense (i.e. forming a grid) 3D or 2D vectors, use the `vector_field` function.

    Arguments:
        vectors: `array_like`. The vectors as (dx, dy, dz) float triples.
        origins: `array_like`. Same-size array of (x, y, z) coordinates of vector origins.
        colors: `array_like`. Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`. Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`. Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        color: `int`. Packed RGB color of the vectors (0xff0000 is red, 0xff is blue) when `colors` is empty and
            origin_color and head_color are not specified.
        use_head: `bool`. Whether vectors should display an arrow head.
        head_size: `float`. The size of the arrow heads.
        labels: `list` of `str`. Captions to display next to the vectors.
        label_size: `float`. Label font size in 'em' HTML units.
        line_width: `float`. Width of the vector segments.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Vectors(
            vectors=vectors,
            origins=origins,
            colors=np.array(colors, np.uint32),
            origin_color=origin_color if origin_color is not None else color,
            head_color=head_color if head_color is not None else color,
            use_head=use_head,
            head_size=head_size,
            labels=labels,
            label_size=label_size,
            line_width=line_width
        ),
        **kwargs
    )


def voxels(voxels, color_map, **kwargs):
    """Create a Voxels drawable for 3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using  kwargs:
        voxels(..., bounds=[0, 300, 0, 400, 0, 500])
    or:
        voxels(..., scaling=[scale_x, scale_y, scale_z]).

    Arguments:
        voxels: `array_like`. 3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`. Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
            The color defined at index i is for voxel value (i+1), e.g.:
            color_map = [0xff, 0x00ff]
            voxels = [[[
                0, # empty voxel
                1, # blue voxel
                2  # red voxel
            ]]]
    kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
    return process_transform_arguments(
        Voxels(voxels=np.array(voxels, np.uint8), color_map=np.array(color_map, np.float32)),
        **kwargs
    )


def vtk_poly_data(poly_data, color=_default_color, color_attribute=None, color_map=basic_color_maps.Rainbow, **kwargs):
    """Create a Mesh drawable from given vtkPolyData.

    This function requires the vtk module (from package VTK) to be installed.

    Arguments:
        poly_data: `vtkPolyData`. Native vtkPolyData geometry.
        color: `int`. Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        color_attribute: `tuple` of (`str`, `float`, `float`). This determines which scalar should be taken as the
            attribute for the color_map, and the color_range for the mesh: (attribute_name, min_value, max_value).
            A VTK mesh can have multiple named attributes in the vertices.
            min_value is the value mapped to 0 in the color_map.
            max_value is the value mapped to 1 in the color_map.
        color_map: `list`. A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        kwargs: `dict`. Dictionary arguments to configure transform and model_matrix."""
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
             color_map=np.array(color_map, np.float32)),
        **kwargs
    )


def plot(*args, **kwargs):
    """Create a K3D Plot widget."""
    return Plot(*args, **kwargs)
