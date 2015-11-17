import base64
import numpy

class Factory(object):
    @classmethod
    def torus_knot(cls, view_matrix=numpy.identity(4)):
        return {
            'type': 'TorusKnot',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': 0xff00ff,
            'knotsNumber': 16,
            'radius': 7,
            'tube': 2
        }

    @classmethod
    def text(cls, string, position=(0, 0, 0), color=0xFFFFFF, font_weight='bold', font_face='Courier New'):
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
    def points(cls, positions, colors, view_matrix=numpy.identity(4), point_size=1.0):
        return {
            'type': 'Points',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'pointSize': point_size,
            'pointsPositions': cls.__to_base64(positions),
            'pointsColors': cls.__to_base64(colors),
        }

    @classmethod
    def line(cls, positions, view_matrix=numpy.identity(4), width=1, color=0xFFFFFF):
        return {
            'type': 'Line',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': color,
            'lineWidth': width,
            'pointsPositions': cls.__to_base64(positions),
        }

    @classmethod
    def surface(cls, heights, width, height, view_matrix=numpy.identity(4), color=0xFFFFFF):
        return {
            'type': 'Surface',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': color,
            'width': width,
            'height': height,
            'heights': list(heights),
        }

    @classmethod
    def marching_cubes(cls, scalars_field, width, height, length, isolation, view_matrix=numpy.identity(4), color=0xFFFFFF):
        return {
            'type': 'MarchingCubes',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'width': width,
            'height': height,
            'length': length,
            'color': color,
            'isolation': isolation,
            'scalarsField': cls.__to_base64(scalars_field),
        }

    @classmethod
    def stl(cls, stl, view_matrix=numpy.identity(4), color=0xFFFFFF):
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
    def vector(cls, origin, vector, view_matrix=numpy.identity(4), label='', color=0xFFFFFF, line_width=1):
        return {
            'type' : 'Vector',
            'modelViewMatrix' : cls.__to_list(view_matrix),
            'origin' : cls.__to_list(origin),
            'vector' : cls.__to_list(vector),
            'color' : color,
            'lineWidth' : line_width,
            'label': label
        }

    @classmethod
    def vector2d(cls, vectors, colors, width, height, view_matrix=numpy.identity(4), use_head=True):
        return {
            'type': 'Vector2D',
            'useHead': use_head,
            'modelViewMatrix': cls.__to_list(view_matrix),
            'width': width,
            'height': height,
            'vectors': cls.__to_base64(vectors),
            'colors': cls.__to_base64(colors)
        }

    @classmethod
    def vector3d(cls, vectors, colors, width, height, length, view_matrix=numpy.identity(4), use_head=True):
        return {
            'type': 'Vector3D',
            'useHead': use_head,
            'modelViewMatrix': cls.__to_list(view_matrix),
            'width': width,
            'height': height,
            'length': length,
            'vectors': cls.__to_base64(vectors),
            'colors': cls.__to_base64(colors)
        }

    @staticmethod
    def __to_list(arg):
        return numpy.array(arg, dtype=numpy.float32).flatten().tolist()

    @staticmethod
    def __to_base64(arg):
        return base64.b64encode(numpy.array(arg, dtype=numpy.float32).data)
