from abc import ABCMeta, abstractmethod
import base64
from functools import partial, reduce
import k3d
import numpy
import sys
from weakref import WeakKeyDictionary

try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

try:
    _strings = (bytes, str, unicode)
except NameError:
    _strings = (bytes, str)

_NoneType = type(None)


class _Attribute(object):
    def __init__(self, expected_type, cast, path):
        self.__expected_type = expected_type
        self.__cast = cast
        self.__path = '/' + path.strip('/')
        self.__values = WeakKeyDictionary()
        self.__transforms = []

    def __set__(self, instance, value):
        for input_type, transform in self.__transforms:
            if isinstance(value, input_type):
                value = transform(value)

        if sys.version_info < (3, 0) and isinstance(value, long):
            value = int(value)

        if isinstance(value, int) and self.__expected_type == float:
            value = float(value)

        if isinstance(value, list) and self.__expected_type == numpy.ndarray:
            value = numpy.array(value, order='C')

        if not isinstance(value, self.__expected_type):
            if type(self.__expected_type) == tuple:
                expected_type_str = ','.join([v.__name__ for v in self.__expected_type])
            else:
                expected_type_str = self.__expected_type.__name__

            raise TypeError('Variable %s. Expected type %s, %s given' % (
                self.__path, expected_type_str, type(value).__name__))

        self.__values[instance] = value

    def __get__(self, instance, cls):
        if instance is None:
            return self

        return self.__get_value(instance)

    def __get_value(self, instance):
        return self.__values[instance] if instance in self.__values else None

    def transform(self, input_type, transform):
        self.__transforms.append((input_type, transform))

        return self

    @property
    def path(self):
        return self.__path

    def get_output(self, instance):
        value = self.__get_value(instance)

        return self.__cast(value) if value is not None else None


def _to_list(ndarray, dtype=numpy.float32):
    return numpy.frombuffer(ndarray.astype(dtype).data, dtype).tolist()


def _to_base64(ndarray, dtype=numpy.float32):
    return base64.b64encode(ndarray.astype(dtype).data).decode(encoding='ascii')


def _to_image_src(url):
    try:
        response = urlopen(url)
    except (IOError, ValueError):
        return url

    content_type = dict(response.info()).get('content-type', 'image/png')

    return 'data:%s;base64,%s' % (content_type, base64.b64encode(response.read()).decode(encoding='ascii'))


def _to_ndarray(data, dtype=numpy.float32):
    return numpy.frombuffer(base64.b64decode(data), dtype=dtype)


class Drawable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __iter__(self):
        pass

    def __add__(self, other):
        return Group(self, other)

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

        super(Drawable, self).__setattr__(name, value)

    @abstractmethod
    def set_plot(self, plot):
        pass


class SingleObject(Drawable):
    __metaclass__ = ABCMeta

    __attributes = None
    __paths = None
    __plot = None

    def __init__(self, **kwargs):
        self.__attributes = tuple(
            key for key, value in self.__class__.__dict__.items() if isinstance(value, _Attribute))
        self.__paths = {value.path: key for key, value in self.__class__.__dict__.items() if
                        isinstance(value, _Attribute)}

        for attr in self.__attributes:
            if attr not in kwargs:
                raise AttributeError('Missing value for attribute "%s"' % attr)

            setattr(self, attr, kwargs.get(attr))

    def __setattr__(self, name, value):
        is_set = getattr(self, name, None) is not None

        super(SingleObject, self).__setattr__(name, value)

        if is_set and name in self.__attributes and self.__plot is not None:
            self.__plot.update(self.id, self.__attr(name))

    @property
    def __dict__(self):
        return {attr: getattr(self, attr) for attr in self.__attributes}

    @property
    def data(self):
        output = {
            'id': self.id,
            'type': self.__class__.__name__,
        }

        for name in self.__attributes:
            attr = self.__attr(name)
            if not self.__is_empty(attr.get('value')):
                self.__set_value(output, attr.get('path'), attr.get('value'))

        return output

    def __attr(self, name):
        attr = getattr(self.__class__, name)

        return {
            'value': attr.get_output(self),
            'path': attr.path,
        }

    @property
    def id(self):
        return id(self)

    def fetch_data(self):
        if self.__plot is not None:
            self.__plot.fetch_data(self)

    def update(self, data):
        for patch in data:
            assert patch['op'] == 'replace'
            assert patch['path'] in self.__paths

            getattr(self.__class__, self.__paths[patch['path']]).__set__(self, patch['value'])

    @staticmethod
    def __set_value(obj, path, value):
        parts = path.strip('/').split('/')
        reduce(lambda obj, part: obj.setdefault(part, {}), parts[:-1], obj).setdefault(parts[-1], value)

    def __iter__(self):
        return (self,).__iter__()

    def set_plot(self, plot):
        assert isinstance(plot, k3d.K3D)

        if self.__plot == plot:
            return False

        if self.__plot is not None:
            raise RuntimeError('Object is already added to another plot')

        self.__plot = plot

        return True

    def unset_plot(self, plot):
        assert isinstance(plot, k3d.K3D)

        if self.__plot is None:
            return False

        if self.__plot != plot:
            raise RuntimeError('Object is added to another plot')

        self.__plot = None

        return True

    @staticmethod
    def __is_empty(value):
        return any((
            value is None,
            hasattr(value, '__len__') and len(value) == 0
        ))


class Group(Drawable):
    __objs = None

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
    color = _Attribute(int, int, 'color')
    line_width = _Attribute((int, float), float, 'lineWidth')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    points_positions = _Attribute(numpy.ndarray, _to_base64, 'pointsPositions').transform(_strings, _to_ndarray)


class MarchingCubes(SingleObject):
    color = _Attribute(int, int, 'color')
    height = _Attribute(int, int, 'height')
    length = _Attribute(int, int, 'length')
    level = _Attribute((int, float), float, 'level')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    scalars_field = _Attribute(numpy.ndarray, partial(_to_base64), 'scalarsField').transform(_strings, _to_ndarray)
    width = _Attribute(int, int, 'width')


class Points(SingleObject):
    color = _Attribute(int, int, 'color')
    shader = _Attribute(str, str, 'shader')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    point_size = _Attribute((int, float), float, 'pointSize')
    points_colors = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint32), 'pointsColors').transform(
        _strings, partial(_to_ndarray, dtype=numpy.uint32))
    points_positions = _Attribute(numpy.ndarray, _to_base64, 'pointsPositions').transform(_strings, _to_ndarray)


class STL(SingleObject):
    color = _Attribute(int, int, 'color')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    stl = _Attribute((bytes, str), str, 'STL')


class Surface(SingleObject):
    color = _Attribute(int, int, 'color')
    height = _Attribute(int, int, 'height')
    heights = _Attribute(numpy.ndarray, partial(_to_base64), 'heights')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    width = _Attribute(int, int, 'width')


class Text(SingleObject):
    color = _Attribute(int, int, 'color')
    size = _Attribute(float, float, 'size')
    font_face = _Attribute(str, str, 'fontOptions/face')
    font_weight = _Attribute(str, str, 'fontOptions/weight')
    font_size = _Attribute(int, int, 'fontOptions/size')
    position = _Attribute(numpy.ndarray, _to_list, 'position')
    text = _Attribute(str, str, 'text')


class Text2d(SingleObject):
    color = _Attribute(int, int, 'color')
    size = _Attribute(float, float, 'size')
    reference_point = _Attribute(str, str, 'referencePoint')
    position = _Attribute(numpy.ndarray, _to_list, 'position')
    text = _Attribute(str, str, 'text')


class Texture(SingleObject):
    image = _Attribute(str, _to_image_src, 'image')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')


class Vectors(SingleObject):
    colors = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint32), 'colors') \
        .transform(_strings, partial(_to_ndarray, dtype=numpy.uint32))
    head_color = _Attribute(int, int, 'headColor')
    labels = _Attribute((list, tuple), tuple, 'labels')
    labels_size = _Attribute(float, float, 'labelsSize')
    head_size = _Attribute(float, float, 'headSize')
    line_width = _Attribute((int, float), float, 'lineWidth')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    origin_color = _Attribute(int, int, 'originColor')
    origins = _Attribute(numpy.ndarray, _to_base64, 'origins').transform(_strings, _to_ndarray)
    vectors = _Attribute(numpy.ndarray, _to_base64, 'vectors').transform(_strings, _to_ndarray)


class VectorFields(SingleObject):
    colors = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint32), 'colors') \
        .transform(_strings, partial(_to_ndarray, dtype=numpy.uint32))
    head_color = _Attribute(int, int, 'headColor')
    height = _Attribute(int, int, 'height')
    length = _Attribute((int, _NoneType), int, 'length')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    origin_color = _Attribute(int, int, 'originColor')
    use_head = _Attribute(bool, bool, 'useHead')
    head_size = _Attribute(float, float, 'headSize')
    vectors = _Attribute(numpy.ndarray, partial(_to_base64), 'vectors').transform(_strings, _to_ndarray)
    width = _Attribute(int, int, 'width')


class Voxels(SingleObject):
    color_map = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint32), 'colorMap') \
        .transform(_strings, partial(_to_ndarray, dtype=numpy.uint32))
    height = _Attribute(int, int, 'height')
    length = _Attribute(int, int, 'length')
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    voxels = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint8), 'voxels') \
        .transform(_strings, partial(_to_ndarray, dtype=numpy.uint8))
    width = _Attribute(int, int, 'width')


class Mesh(SingleObject):
    model_matrix = _Attribute(numpy.ndarray, _to_list, 'modelMatrix')
    vertices = _Attribute(numpy.ndarray, _to_base64, 'vertices').transform(_strings, _to_ndarray)
    indices = _Attribute(numpy.ndarray, partial(_to_base64, dtype=numpy.uint32), 'indices') \
        .transform(_strings, partial(_to_ndarray, dtype=numpy.uint32))
    color = _Attribute(int, int, 'color')
    vertex_scalars = _Attribute(numpy.ndarray, _to_base64, 'vertex_scalars').transform(_strings, _to_ndarray)
    color_map = _Attribute(numpy.ndarray, _to_base64, 'color_map').transform(_strings, _to_ndarray)
    color_range = _Attribute(numpy.ndarray, _to_base64, 'color_range').transform(_strings, _to_ndarray)
