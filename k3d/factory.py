import base64
import numpy

class Factory(object):
    DEFAULT_COLOR = 0x0000FF

    @classmethod
    def torus_knot(cls, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4)):
        return {
            'type': 'TorusKnot',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'color': 0xff00ff,
            'knotsNumber': 16,
            'radius': 7,
            'tube': 2
        }

    @classmethod
    def text(cls, string, position=(0, 0, 0), color=DEFAULT_COLOR, font_weight='bold', font_face='Courier New'):
        return {
            'type': 'Text',
            'position': cls.__to_list(position),
            'text': string,
            'color': color,
            'fontOptions': {
                'face': font_face,
                'weight': font_weight
            }
        }

    @classmethod
    def points(cls, positions, colors=(), color=DEFAULT_COLOR, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), point_size=1.0):
        return dict(
            [
                ('type', 'Points'),
                ('modelViewMatrix', cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax)),
                ('pointSize', point_size),
                ('pointsPositions', cls.__to_base64(positions))
            ] + [
                ('pointsColors', cls.__to_base64(colors, numpy.uint32)) if len(colors) > 0 else ('color', color)
            ]
        )

    @classmethod
    def line(cls, positions, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), width=1, color=DEFAULT_COLOR):
        return {
            'type': 'Line',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'color': color,
            'lineWidth': width,
            'pointsPositions': cls.__to_base64(positions),
        }

    @classmethod
    def surface(cls, heights, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), width=None, height=None, color=DEFAULT_COLOR):
        width, height = cls.__get_dimensions(numpy.shape(heights), width, height)

        return {
            'type': 'Surface',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'color': color,
            'width': width,
            'height': height,
            'heights': cls.__to_base64(heights),
        }

    @classmethod
    def marching_cubes(cls, scalars_field, level, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), width=None, height=None, length=None, color=DEFAULT_COLOR):
        width, height, length = cls.__get_dimensions(numpy.shape(scalars_field), width, height, length)

        return {
            'type': 'MarchingCubes',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'color': color,
            'level': level,
            'scalarsField': cls.__to_base64(scalars_field),
        }

    @classmethod
    def stl(cls, stl, view_matrix=numpy.identity(4), color=DEFAULT_COLOR):
        return {
            'type': 'STL',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': color,
            'STL': stl,
        }

    @classmethod
    def stl_load(cls, filename, view_matrix=numpy.identity(4)):
        with open(filename) as stl:
            return cls.stl(stl.read(), view_matrix)

    @classmethod
    def vectors(cls, origins, vectors, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), labels=(), colors=(), color=DEFAULT_COLOR, line_width=1):
        return dict(
            [
                ('type', 'Vectors'),
                ('modelViewMatrix', cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax)),
                ('origins', cls.__to_base64(origins)),
                ('vectors', cls.__to_base64(vectors)),
                ('lineWidth', line_width),
            ] + (
                [('labels', labels)] if len(labels) > 0 else []
            ) + (
                [('colors', cls.__to_base64(colors, numpy.uint32))] if len(colors) > 0 else [('originColor', color), ('headColor', color)]
            )
        )

    @classmethod
    def vectors_fields(cls, vectors, colors=(), color=DEFAULT_COLOR, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), width=None, height=None, length=None, use_head=True):
        shape = numpy.shape(vectors)
        width, height, length = cls.__get_dimensions(shape[:-1] + (None,), width, height, length)

        cls.__validate_vectors_size(length, vector_size=shape[-1])

        return dict(
            [
                ('type', 'VectorsFields'),
                ('useHead', use_head),
                ('modelViewMatrix', cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax)),
                ('vectors', cls.__to_base64(vectors)),
                ('width', width),
                ('height', height),
            ] + (
                [('colors', cls.__to_base64(colors, numpy.uint32))] if len(colors) > 0 else [('originColor', color), ('headColor', color)]
            ) + (
                [('length', length)] if length is not None else []
            )
        )

    @staticmethod
    def __validate_vectors_size(length, vector_size):
        expected_vectors_size = 2 if length is None else 3

        if vector_size is not expected_vectors_size:
            raise TypeError('Invalid vectors size: expected %d, %d given' % (expected_vectors_size, vector_size))

    @classmethod
    def texture(cls, image, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4)):
        return {
            'type': 'Texture',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'image': image,
        }

    @classmethod
    def voxels(cls, voxels, color_map, xmin=-.5, xmax=.5, ymin=-.5, ymax=.5, zmin=-.5, zmax=.5, view_matrix=numpy.identity(4), width=None, height=None, length=None):
        width, height, length = cls.__get_dimensions(numpy.shape(voxels), width, height, length)

        return {
            'type': 'Voxels',
            'modelViewMatrix': cls.__get_view_matrix_list(view_matrix, xmin, xmax, ymin, ymax, zmin, zmax),
            'width': width,
            'height': height,
            'length': length,
            'colorMap': cls.__to_list(color_map),
            'voxels': cls.__to_base64(voxels, numpy.uint8),
        }

    @staticmethod
    def __to_list(array_like, dtype=numpy.float32):
        return numpy.frombuffer(numpy.array(array_like, dtype, order='F').data, dtype).tolist()

    @staticmethod
    def __to_base64(array_like, dtype=numpy.float32):
        return base64.b64encode(numpy.array(array_like, dtype, order='F').data)

    @staticmethod
    def __get_dimensions(shape, *dimensions):
        return dimensions if len(shape) < len(dimensions) else [val or shape[i] for i, val in enumerate(dimensions)]

    @classmethod
    def __get_view_matrix_list(cls, view_matrix, xmin, xmax, ymin, ymax, zmin, zmax):
        view_matrix = numpy.array(view_matrix, numpy.float32, order='C').reshape(4, -1)

        if view_matrix.shape != (4, 4):
            raise ValueError('view_matrix: expected 4x4 matrix, %s given' % 'x'.join(str(i) for i in view_matrix.shape))

        return numpy.dot(cls.__get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax), view_matrix).flatten().tolist()

    @staticmethod
    def __get_base_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
        for name, value in locals().items():
            try:
                float(value)
            except (TypeError, ValueError):
                raise TypeError('%s: expected float, %s given' % (name, type(value).__name__))

        matrix = numpy.diagflat(numpy.array((xmax - xmin, ymax - ymin, zmax - zmin, 1.0), numpy.float32, order='C'))
        matrix[0:3, 3] = ((xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2)

        return matrix
