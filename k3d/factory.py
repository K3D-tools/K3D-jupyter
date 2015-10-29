from itertools import chain

class Factory:
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
    def text(cls, view_matrix, string, color = 0xFFFFFF, font_weight = 'bold', font_face = 'Courier New'):
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
    def points(cls, view_matrix, positions, colors, point_size = 1.0):
        return {
            'type': 'Points',
            'modelViewMatrix': cls.__to_list(view_matrix),
            'pointSize': point_size,
            'pointsPositions': cls.__to_list(positions),
            'pointsColors': cls.__to_list(colors),
        }

    @classmethod
    def line(cls, view_matrix, positions, width=1, color=0xFFFFFF):
        return {
            'type': 'Line',
            'modelViewMatrix': cls.__matrix_to_list(view_matrix),
            'color': color,
            'lineWidth': width,
            'pointsPositions': cls.__matrix_to_list(positions),
        }

    @classmethod
    def surface(cls, view_matrix, heights, resolution):
        return {
            'type': 'Surface',
            'modelViewMatrix': cls.__matrix_to_list(view_matrix),
            'heights': list(heights),
            'resolution': resolution,
        }

    @classmethod
    def __to_list(cls, arg):
        if not hasattr(arg, '__iter__'):
            return [arg]

        return sum(map(cls.__to_list, list(arg)), [])
