from abc import ABCMeta, abstractmethod
import base64
import k3d
import numpy


class Drawable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __iter__(self):
        pass

    def __add__(self, other):
        return Group(self, other)

    @abstractmethod
    def set_plot(self, plot):
        pass


class SingleObject(Drawable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, obj):
        assert isinstance(obj, dict)

        self.__obj = obj
        self.__plot = None

    def __iter__(self):
        return (self,).__iter__()

    @property
    def __dict__(self):
        items = self.__obj.items() + [('type', self.__class__.__name__)]

        return {key: value for key, value in items if not self.__is_empty(value)}

    def set_plot(self, plot):
        assert isinstance(plot, k3d.K3D)

        if self.__plot == plot:
            return False

        if self.__plot is not None:
            raise RuntimeError('Object is already added to another plot')

        self.__plot = plot

        return True

    @staticmethod
    def _to_list(ndarray, dtype=numpy.float32):
        assert isinstance(ndarray, numpy.ndarray)
        assert ndarray.dtype == dtype

        return numpy.frombuffer(ndarray.data, ndarray.dtype).tolist()

    @staticmethod
    def _to_base64(ndarray, dtype=numpy.float32):
        assert isinstance(ndarray, numpy.ndarray)
        assert ndarray.dtype == dtype

        return base64.b64encode(ndarray.data)

    @staticmethod
    def __is_empty(value):
        return any((
            value is None,
            hasattr(value, '__len__') and len(value) == 0
        ))


class Group(Drawable):
    def __init__(self, *args):
        self.__objs = tuple(self.__assert_drawable(drawable) for drawables in args for drawable in drawables)

    def __iter__(self):
        return self.__objs.__iter__()

    def set_plot(self, plot):
        return all(obj.set_plot(plot) for obj in self.__objs)

    @staticmethod
    def __assert_drawable(arg):
        assert isinstance(arg, Drawable)

        return arg


class Line(SingleObject):
    def __init__(self, color, line_width, model_view_matrix, points_positions):
        super(Line, self).__init__({
            'color': int(color),
            'lineWidth': float(line_width),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'pointsPositions': self._to_base64(points_positions),
        })


class MarchingCubes(SingleObject):
    def __init__(self, color, height, length, level, model_view_matrix, scalars_field, width):
        super(MarchingCubes, self).__init__({
            'color': int(color),
            'height': int(height),
            'length': int(length),
            'level': float(level),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'scalarsField': self._to_base64(scalars_field),
            'width': int(width),
        })


class Points(SingleObject):
    def __init__(self, color, model_view_matrix, point_size, points_colors, points_positions):
        super(Points, self).__init__({
            'color': int(color),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'pointSize': float(point_size),
            'pointsColors': self._to_base64(points_colors, numpy.uint32),
            'pointsPositions': self._to_base64(points_positions),
        })


class STL(SingleObject):
    def __init__(self, color, model_view_matrix, stl):
        super(STL, self).__init__({
            'color': int(color),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'STL': bytes(stl),
        })


class Surface(SingleObject):
    def __init__(self, color, height, heights, model_view_matrix, width):
        super(Surface, self).__init__({
            'color': int(color),
            'height': int(height),
            'heights': self._to_base64(heights),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'width': int(width),
        })


class Text(SingleObject):
    def __init__(self, color, font_face, font_weight, position, text):
        super(Text, self).__init__({
            'color': int(color),
            'fontOptions': {
                'face': str(font_face),
                'weight': str(font_weight),
            },
            'position': self._to_list(position),
            'text': str(text),
        })


class Texture(SingleObject):
    def __init__(self, image, model_view_matrix):
        super(Texture, self).__init__({
            'image': str(image),
            'modelViewMatrix': self._to_list(model_view_matrix),
        })


class Vectors(SingleObject):
    def __init__(self, colors, color, labels, line_width, model_view_matrix, origins, vectors):
        super(Vectors, self).__init__({
            'colors': self._to_base64(colors, numpy.uint32),
            'headColor': int(color),
            'labels': list(labels),
            'lineWidth': float(line_width),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'originColor': int(color),
            'origins': self._to_base64(origins),
            'vectors': self._to_base64(vectors),
        })


class VectorsFields(SingleObject):
    def __init__(self, colors, color, height, length, model_view_matrix, use_head, vectors, width):
        super(VectorsFields, self).__init__({
            'colors': self._to_base64(colors, numpy.uint32),
            'headColor': int(color),
            'height': int(height),
            'length': int(length) if length is not None else None,
            'modelViewMatrix': self._to_list(model_view_matrix),
            'originColor': int(color),
            'useHead': bool(use_head),
            'vectors': self._to_base64(vectors),
            'width': int(width),
        })


class Voxels(SingleObject):
    def __init__(self, color_map, height, length, model_view_matrix, voxels, width):
        super(Voxels, self).__init__({
            'colorMap': self._to_list(color_map, numpy.uint32),
            'height': int(height),
            'length': int(length),
            'modelViewMatrix': self._to_list(model_view_matrix),
            'voxels': self._to_base64(voxels, numpy.uint8),
            'width': int(width),
        })
