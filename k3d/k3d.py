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
from .helpers import get_model_matrix, get_dimensions, validate_vectors_size
from .objects import (Line, MarchingCubes, Mesh, Points, STL, Surface, Text, Texture, TextureText, VectorField,
                      Vectors, Voxels)

_default_color = 0x0000FF


def line(vertices, color=_default_color, width=1,
         xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, model_matrix=np.identity(4)):
    """Create a Line drawable for plotting segments and polylines.

    Arguments:
        vertices: `array_like`. Array with (x, y, z) coordinates of segment endpoints.
        color: `int`. Packed RGB color of the lines (0xff0000 is red, 0xff is blue).
        width: `float`. Thickness of the lines.
        model_matrix: `array_like`. 4x4 model transform matrix."""
    return Line(vertices=np.array(vertices, np.float32), color=color, width=width,
                model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax))


def marching_cubes(scalar_field, level, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
                   model_matrix=np.identity(4), width=None, height=None, length=None, color=_default_color):
    """Create a MarchingCubes drawable.

    Plot an isosurface of a scalar field obtained through a Marching Cubes algorithm."""
    length, height, width = get_dimensions(np.shape(scalar_field), length, height, width)

    return MarchingCubes(model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
                         width=width, height=height, length=length,
                         color=color,
                         level=level,
                         scalar_field=np.array(scalar_field, np.float32))


def mesh(vertices, indices, attribute=[], color_range=[], color_map=[], model_matrix=np.identity(4),
         color=_default_color):
    """Create a Mesh drawable."""
    return Mesh(model_matrix=model_matrix,
                vertices=np.array(vertices, np.float32),
                indices=np.array(indices, np.uint32),
                color=color,
                attribute=np.array(attribute, np.float32),
                color_range=color_range, color_map=np.array(color_map, np.float32))


def points(positions, colors=[], color=_default_color, model_matrix=np.identity(4), point_size=1.0,
           shader='3dSpecular'):
    """Create a Points drawable representing a point cloud.

    Arguments:
        positions: `array_like`. Array with (x, y, z) coordinates of the points.
        colors: `array_like`. Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`. Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        point_size: `float`. Diameter of the balls representing the points in 3D space.
        shader: `str`. Display style (name of the shader used) of the points.
            Legal values are: `flat`, `3d` and `3dSpecular`.
        model_matrix: `array_like`. 4x4 model transform matrix."""
    return Points(positions=np.array(positions, np.float32), colors=np.array(colors, np.float32),
                  color=color, point_size=point_size, shader=shader,
                  model_matrix=get_model_matrix(model_matrix))


def stl(stl, model_matrix=np.identity(4), color=_default_color):
    """Create an STL drawable."""
    return STL(model_matrix=model_matrix,
               color=color,
               text=stl if isinstance(stl, six.string_types) else None,
               binary=np.array(stl) if not isinstance(stl, six.string_types) else np.array([]))


def surface(heights, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, model_matrix=np.identity(4), width=None,
            height=None, color=_default_color):
    """Create a Surface drawable."""
    height, width = get_dimensions(np.shape(heights), height, width)

    return Surface(model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax),
                   color=color,
                   width=width, height=height,
                   heights=np.array(heights, np.float32))


def text(text, position=[0, 0, 0], color=_default_color, size=1.0, reference_point='lb'):
    """Create a Text drawable."""
    return Text(position=position, reference_point=reference_point, text=text, size=size, color=color)


def texture(binary, file_format, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
            model_matrix=np.identity(4)):
    """Create a Texture drawable."""
    return Texture(model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
                   binary=binary,
                   file_format=file_format)


def texture_text(text, position=[0, 0, 0], color=_default_color, font_weight=400, font_face='Courier New',
                 font_size=68, size=1.0):
    """Create a TextureText drawable."""
    return TextureText(position=position, text=text, color=color, size=size,
                       font_face=font_face, font_size=font_size, font_weight=font_weight)


def vector_field(vectors, colors=[], color=_default_color, width=None, height=None, length=None,
                 use_head=True, head_size=1.0, head_color=None, origin_color=None,
                 xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, model_matrix=np.identity(4)):
    """Create a VectorField drawable.

    Arguments:
        vectors: `array_like`. Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
        colors: `array_like`. Same-length array of (`int`) packed RGB color of the vectors (0xff0000 is red, 0xff is blue).
        color: `int`. Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        width: `int`. Width of the vector field when `vectors` is a flat array.
        height: `int`. Height of the vector field when `vectors` is a flat array.
        length: `int`. Length of the vector field when `vectors` is a flat array.
        use_head: `bool`.
        head_size: `float`.
        head_color: `int`. Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue).
        origin_color: `int`. Packed RGB color of the vector origins (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`. 4x4 model transform matrix."""
    shape = np.shape(vectors)

    if len(shape[:-1]) < 3:
        shape = (None,) + shape

    length, height, width = get_dimensions(shape[:-1], length, height, width)

    validate_vectors_size(length, vector_size=shape[-1])
    return VectorField(vectors=np.array(vectors, np.float32),
                       colors=np.array(colors, np.uint32),
                       width=width, height=height, length=length,
                       use_head=use_head,
                       head_size=head_size,
                       head_color=head_color if head_color is not None else color,
                       origin_color=origin_color if origin_color is not None else color,
                       model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax))


def vectors(origins, vectors, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
            model_matrix=np.identity(4), use_head=True, labels=[], colors=[], color=_default_color, line_width=1,
            label_size=1.0, head_size=1.0, head_color=None, origin_color=None, scale=1.0):
    """Create a Vectors drawable."""
    return Vectors(model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
                   use_head=use_head,
                   origins=np.array(origins, np.float32),
                   vectors=np.array(vectors, np.float32),
                   line_width=line_width,
                   labels=labels,
                   label_size=label_size,
                   head_size=head_size,
                   scale=scale,
                   colors=np.array(colors, np.uint32),
                   head_color=head_color if head_color is not None else color,
                   origin_color=origin_color if origin_color is not None else color)


def voxels(voxels, color_map, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
           model_matrix=np.identity(4), width=None, height=None, length=None):
    length, height, width = get_dimensions(np.shape(voxels), length, height, width)
    """Create a Voxels drawable."""
    return Voxels(model_matrix=get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
                  width=width, height=height, length=length,
                  color_map=np.array(color_map, np.float32),
                  voxels=np.array(voxels, np.uint8))


def vtk_poly_data(poly_data, model_matrix=np.identity(4), color=_default_color, color_attribute=None,
                  color_map=basic_color_maps.Rainbow):
    """Create a Mesh drawable from given vtkPolyData."""
    if vtk is None:
        raise RuntimeError('Module vtk is not available')

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

    return Mesh(model_matrix=model_matrix,
                vertices=np.array(vertices, np.float32),
                indices=np.array(indices, np.uint32),
                color=color,
                attribute=np.array(attribute, np.float32),
                color_range=color_range,
                color_map=np.array(color_map, np.float32))


def plot(*args, **kwargs):
    """Create a K3D Plot widget."""
    return Plot(*args, **kwargs)
