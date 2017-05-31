# optional dependency
try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    vtk = None
    numpy_support = None

from .colormaps.basic_color_maps import basic_color_maps
from .objects import (Line, Mesh, MarchingCubes, Points, Surface, STL, Text, Text2d, Texture, Vectors, VectorFields,
                      Voxels)
import numpy as np


class Factory(object):
    DEFAULT_COLOR = 0x0000FF

    @classmethod
    def text(cls, string, position=(0, 0, 0), color=DEFAULT_COLOR, font_weight='bold', font_face='Courier New',
             font_size=68, size=1.0):
        return Text(**{
            'position': cls.__to_ndarray(position),
            'text': string,
            'color': color,
            'size': size,
            'font_face': font_face,
            'font_size': font_size,
            'font_weight': font_weight,
        })

    @classmethod
    def text2d(cls, string, position=(0, 0, 0), color=DEFAULT_COLOR, size=1.0, reference_point='lb'):
        return Text2d(**{
            'position': cls.__to_ndarray(position),
            'reference_point': reference_point,
            'text': string,
            'size': size,
            'color': color,
        })

    @classmethod
    def points(cls, positions, colors=(), color=DEFAULT_COLOR, model_matrix=np.identity(4), point_size=1.0,
               shader='3dSpecular'):
        return Points(**{
            'model_matrix': cls.__get_model_matrix(model_matrix),
            'point_size': point_size,
            'point_positions': cls.__to_ndarray(positions),
            'point_colors': cls.__to_ndarray(colors, np.uint32),
            'color': color,
            'shader': shader,
        })

    @classmethod
    def line(cls, positions, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, model_matrix=np.identity(4),
             width=1, color=DEFAULT_COLOR):
        return Line(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'color': color,
            'line_width': width,
            'point_positions': cls.__to_ndarray(positions),
        })

    @classmethod
    def surface(cls, heights, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, model_matrix=np.identity(4), width=None,
                height=None, color=DEFAULT_COLOR):
        height, width = cls.__get_dimensions(np.shape(heights), height, width)

        return Surface(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax),
            'color': color,
            'width': width,
            'height': height,
            'heights': cls.__to_ndarray(heights),
        })

    @classmethod
    def marching_cubes(cls, scalar_field, level, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
                       model_matrix=np.identity(4), width=None, height=None, length=None, color=DEFAULT_COLOR):
        length, height, width = cls.__get_dimensions(np.shape(scalar_field), length, height, width)

        return MarchingCubes(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'color': color,
            'level': level,
            'scalar_field': cls.__to_ndarray(scalar_field),
        })

    @classmethod
    def mesh(cls, vertices, indices, vertex_scalars=(), color_range=(), color_map=(), model_matrix=np.identity(4),
             color=DEFAULT_COLOR):
        return Mesh(**{
            'model_matrix': cls.__get_model_matrix(model_matrix),
            'vertices': cls.__to_ndarray(vertices),
            'indices': cls.__to_ndarray(indices),
            'color': color,
            'vertex_scalars': cls.__to_ndarray(vertex_scalars),
            'color_range': cls.__to_ndarray(color_range),
            'color_map': cls.__to_ndarray(color_map)
        })

    @classmethod
    def vtk_poly_data(cls, poly_data, model_matrix=np.identity(4), color=DEFAULT_COLOR, color_attribute=None,
                      color_map=basic_color_maps.Rainbow):
        if poly_data.GetPolys().GetMaxCellSize() > 3:
            cutTriangles = vtk.vtkTriangleFilter()
            cutTriangles.SetInputData(poly_data)
            cutTriangles.Update()
            poly_data = cutTriangles.GetOutput()

        if color_attribute is not None:
            vertex_scalars = numpy_support.vtk_to_numpy(poly_data.GetPointData().GetArray(color_attribute[0]))
            color_range = color_attribute[1:3]
        else:
            vertex_scalars = ()
            color_range = ()

        vertices = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
        indices = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

        return Mesh(**{
            'model_matrix': cls.__get_model_matrix(model_matrix),
            'vertices': cls.__to_ndarray(vertices),
            'indices': cls.__to_ndarray(indices, np.uint32),
            'color': color,
            'vertex_scalars': cls.__to_ndarray(vertex_scalars),
            'color_range': cls.__to_ndarray(color_range),
            'color_map': cls.__to_ndarray(color_map)
        })

    @classmethod
    def stl(cls, stl, model_matrix=np.identity(4), color=DEFAULT_COLOR):
        return STL(**{
            'model_matrix': cls.__get_model_matrix(model_matrix),
            'color': color,
            'stl': stl
        })

    @classmethod
    def stl_load(cls, filename, model_matrix=np.identity(4)):
        with open(filename) as stl:
            return cls.stl(stl.read(), model_matrix)

    @classmethod
    def vectors(cls, origins, vectors, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
                model_matrix=np.identity(4), labels=(), colors=(), color=DEFAULT_COLOR, line_width=1,
                label_size=1.0, head_size=1.0, head_color=None, origin_color=None):
        return Vectors(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'origins': cls.__to_ndarray(origins),
            'vectors': cls.__to_ndarray(vectors),
            'line_width': line_width,
            'labels': labels,
            'label_size': label_size,
            'head_size': head_size,
            'colors': cls.__to_ndarray(colors, np.uint32),
            'head_color': head_color if head_color is not None else color,
            'origin_color': origin_color if origin_color is not None else color,
        })

    @classmethod
    def vector_fields(cls, vectors, colors=(), color=DEFAULT_COLOR, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5,
                      zmax=.5, model_matrix=np.identity(4), width=None, height=None, length=None, use_head=True,
                      head_color=None, head_size=1.0, origin_color=None):
        shape = np.shape(vectors)

        if len(shape[:-1]) < 3:
            shape = (None,) + shape

        length, height, width = cls.__get_dimensions(shape[:-1], length, height, width)

        cls.__validate_vectors_size(length, vector_size=shape[-1])

        return VectorFields(**{
            'use_head': use_head,
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'vectors': cls.__to_ndarray(vectors),
            'width': width,
            'height': height,
            'head_size': head_size,
            'colors': cls.__to_ndarray(colors, np.uint32),
            'head_color': head_color if head_color is not None else color,
            'origin_color': origin_color if head_color is not None else color,
            'length': length,
        })

    @staticmethod
    def __validate_vectors_size(length, vector_size):
        expected_vectors_size = 2 if length is None else 3

        if vector_size is not expected_vectors_size:
            raise TypeError('Invalid vectors size: expected %d, %d given' % (expected_vectors_size, vector_size))

    @classmethod
    def texture(cls, image, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, model_matrix=np.identity(4)):
        return Texture(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'image': image,
        })

    @classmethod
    def voxels(cls, voxels, color_map, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
               model_matrix=np.identity(4), width=None, height=None, length=None):
        length, height, width = cls.__get_dimensions(np.shape(voxels), length, height, width)

        return Voxels(**{
            'model_matrix': cls.__get_model_matrix(model_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'color_map': cls.__to_ndarray(color_map, np.uint32),
            'voxels': cls.__to_ndarray(voxels, np.uint8),
        })

    @staticmethod
    def __to_ndarray(array_like, dtype=np.float32):
        return np.array(array_like, dtype, order='C')

    @staticmethod
    def __get_dimensions(shape, *dimensions):
        return dimensions if len(shape) < len(dimensions) else [val or shape[i] for i, val in enumerate(dimensions)]

    @classmethod
    def __get_model_matrix(cls, model_matrix, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5):
        model_matrix = np.array(model_matrix, np.float32, order='C').reshape(4, -1)

        if model_matrix.shape != (4, 4):
            raise ValueError(
                'model_matrix: expected 4x4 matrix, %s given' % 'x'.join(str(i) for i in model_matrix.shape))

        return np.dot(cls.__get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax), model_matrix)

    @staticmethod
    def __get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
        for name, value in locals().items():
            try:
                float(value)
            except (TypeError, ValueError):
                raise TypeError('%s: expected float, %s given' % (name, type(value).__name__))

        matrix = np.diagflat(np.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), np.float32, order='C'))
        matrix[0:3, 3] = ((xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0)

        return matrix
