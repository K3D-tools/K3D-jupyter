import numpy

# optional dependency
try:
    import vtk
    from vtk.util import numpy_support
except:
    vtk = None
    numpy_support = None

from .objects import *
from .colormaps import colorMaps


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
    def points(cls, positions, colors=(), color=DEFAULT_COLOR, view_matrix=numpy.identity(4), point_size=1.0,
               shader='3dSpecular'):
        return Points(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix),
            'point_size': point_size,
            'points_positions': cls.__to_ndarray(positions),
            'points_colors': cls.__to_ndarray(colors, numpy.uint32),
            'color': color,
            'shader': shader,
        })

    @classmethod
    def line(cls, positions, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4),
             width=1, color=DEFAULT_COLOR):
        return Line(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'color': color,
            'line_width': width,
            'points_positions': cls.__to_ndarray(positions),
        })

    @classmethod
    def surface(cls, heights, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, view_matrix=numpy.identity(4), width=None,
                height=None, color=DEFAULT_COLOR):
        height, width = cls.__get_dimensions(numpy.shape(heights), height, width)

        return Surface(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax),
            'color': color,
            'width': width,
            'height': height,
            'heights': cls.__to_ndarray(heights),
        })

    @classmethod
    def marching_cubes(cls, scalars_field, level, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
                       view_matrix=numpy.identity(4), width=None, height=None, length=None, color=DEFAULT_COLOR):
        length, height, width = cls.__get_dimensions(numpy.shape(scalars_field), length, height, width)

        return MarchingCubes(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'color': color,
            'level': level,
            'scalars_field': cls.__to_ndarray(scalars_field),
        })

    @classmethod
    def mesh(cls, vertices, indices, view_matrix=numpy.identity(4), color=DEFAULT_COLOR):
        return Mesh(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix),
            'vertices': cls.__to_ndarray(vertices),
            'indices': cls.__to_ndarray(indices),
            'color': color
        })

    @classmethod
    def vtkPolyData(cls, polyData, view_matrix=numpy.identity(4), color=DEFAULT_COLOR, color_attribute=None,
                    color_map=colorMaps.Rainbow):

        if polyData.GetPolys().GetMaxCellSize() > 3:
            cutTriangles = vtk.vtkTriangleFilter()
            cutTriangles.SetInputData(polyData)
            cutTriangles.Update()
            polyData = cutTriangles.GetOutput()

        if color_attribute is not None:
            vertex_scalars = numpy_support.vtk_to_numpy(polyData.GetPointData().GetArray(color_attribute[0]))
            color_range = color_attribute[1:3]
        else:
            vertex_scalars = ()
            color_range = ()

        vertices = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
        indices = numpy_support.vtk_to_numpy(polyData.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

        return Mesh(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix),
            'vertices': cls.__to_ndarray(vertices),
            'indices': cls.__to_ndarray(indices, numpy.uint32),
            'color': color,
            'vertex_scalars': cls.__to_ndarray(vertex_scalars),
            'color_range': cls.__to_ndarray(color_range),
            'color_map': cls.__to_ndarray(color_map)
        })

    @classmethod
    def stl(cls, stl, view_matrix=numpy.identity(4), color=DEFAULT_COLOR):
        return STL(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix),
            'color': color,
            'stl': stl
        })

    @classmethod
    def stl_load(cls, filename, view_matrix=numpy.identity(4)):
        with open(filename) as stl:
            return cls.stl(stl.read(), view_matrix)

    @classmethod
    def vectors(cls, origins, vectors, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
                view_matrix=numpy.identity(4), labels=(), colors=(), color=DEFAULT_COLOR, line_width=1, labels_size=1.0,
                head_color=None,
                origin_color=None):
        return Vectors(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'origins': cls.__to_ndarray(origins),
            'vectors': cls.__to_ndarray(vectors),
            'line_width': line_width,
            'labels': labels,
            'labels_size': labels_size,
            'colors': cls.__to_ndarray(colors, numpy.uint32),
            'head_color': head_color if head_color is not None else color,
            'origin_color': origin_color if origin_color is not None else color,
        })

    @classmethod
    def vectors_fields(cls, vectors, colors=(), color=DEFAULT_COLOR, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5,
                       zmax=.5, view_matrix=numpy.identity(4), width=None, height=None, length=None, use_head=True,
                       head_color=None, origin_color=None):
        shape = numpy.shape(vectors)

        if len(shape[:-1]) < 3:
            shape = (None,) + shape

        length, height, width = cls.__get_dimensions(shape[:-1], length, height, width)

        cls.__validate_vectors_size(length, vector_size=shape[-1])

        return VectorsFields(**{
            'use_head': use_head,
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'vectors': cls.__to_ndarray(vectors),
            'width': width,
            'height': height,
            'colors': cls.__to_ndarray(colors, numpy.uint32),
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
    def texture(cls, image, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4)):
        return Texture(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'image': image,
        })

    @classmethod
    def voxels(cls, voxels, color_map, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5,
               view_matrix=numpy.identity(4), width=None, height=None, length=None):
        length, height, width = cls.__get_dimensions(numpy.shape(voxels), length, height, width)

        return Voxels(**{
            'model_view_matrix': cls.__get_view_matrix(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'color_map': cls.__to_ndarray(color_map, numpy.uint32),
            'voxels': cls.__to_ndarray(voxels, numpy.uint8),
        })

    @staticmethod
    def __to_ndarray(array_like, dtype=numpy.float32):
        return numpy.array(array_like, dtype, order='C')

    @staticmethod
    def __get_dimensions(shape, *dimensions):
        return dimensions if len(shape) < len(dimensions) else [val or shape[i] for i, val in enumerate(dimensions)]

    @classmethod
    def __get_view_matrix(cls, view_matrix, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5):
        view_matrix = numpy.array(view_matrix, numpy.float32, order='C').reshape(4, -1)

        if view_matrix.shape != (4, 4):
            raise ValueError('view_matrix: expected 4x4 matrix, %s given' % 'x'.join(str(i) for i in view_matrix.shape))

        return numpy.dot(cls.__get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax), view_matrix)

    @staticmethod
    def __get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
        for name, value in locals().items():
            try:
                float(value)
            except (TypeError, ValueError):
                raise TypeError('%s: expected float, %s given' % (name, type(value).__name__))

        matrix = numpy.diagflat(numpy.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), numpy.float32, order='C'))
        matrix[0:3, 3] = ((xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0)

        return matrix
