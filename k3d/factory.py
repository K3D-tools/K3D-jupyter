import base64, numpy

class Factory(object):
    @classmethod
    def torus_knot(cls, view_matrix):
        return {
            'type': 'TorusKnot',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': 0xff00ff,
            'knotsNumber': 16,
            'radius': 7,
            'tube': 2
        }

    @classmethod
    def text(cls, view_matrix, string, color=0xFFFFFF, font_weight='bold', font_face='Courier New'):
        return {
            'type': 'Text',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'text': string,
            'color': color,
            'fontOptions': {
                'face': font_face,
                'weight': font_weight
            }
        }

    @classmethod
    def points(cls, view_matrix, positions, colors, point_size=1.0):
        return {
            'type': 'Points',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'pointSize': point_size,
            'pointsPositions': cls.__to_base64(positions),
            'pointsColors': cls.__to_base64(colors),
        }

    @classmethod
    def line(cls, view_matrix, positions, width=1, color=0xFFFFFF):
        return {
            'type': 'Line',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': color,
            'lineWidth': width,
            'pointsPositions': cls.__to_base64(positions),
        }

    @classmethod
    def surface(cls, view_matrix, heights, resolution):
        return {
            'type': 'Surface',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'heights': list(heights),
            'resolution': resolution,
        }

    @classmethod
    def marching_cubes(cls, view_matrix, scalars_field, resolution, isolation, color=0xFFFFFF):
        return {
            'type': 'MarchingCubes',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'resolution': resolution,
            'color': color,
            'isolation': isolation,
            'scalarsField': cls.__to_base64(scalars_field),
        }

    @classmethod
    def stl(cls, view_matrix, stl, color=0xFFFFFF):
        return {
            'type': 'STL',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'color': color,
            'STL': stl,
        }

    @classmethod
    def stl_load(cls, filename, view_matrix=numpy.identity(4)):
        with open(filename) as stl:
            return cls.stl(view_matrix, stl.read())

    @staticmethod
    def __to_list(arg):
        return numpy.array(arg, dtype=numpy.float32).flatten().tolist()

    @staticmethod
    def __to_base64(arg):
        return base64.b64encode(numpy.array(arg, dtype=numpy.float32).data)
