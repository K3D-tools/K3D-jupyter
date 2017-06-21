import base64
import ipywidgets as widgets
from traitlets import Unicode, Int, Float, List, Bool, Bytes
from traitlets import validate, TraitError
from traittypes import Array
from .helpers import array_serialization
from ._version import __version__

try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen


def _to_image_src(url):
    try:
        response = urlopen(url)
    except (IOError, ValueError):
        return url

    content_type = dict(response.info()).get('content-type', 'image/png')

    return 'data:%s;base64,%s' % (content_type, base64.b64encode(response.read()).decode(encoding='ascii'))


class Drawable(widgets.CoreWidget):
    """
    Base class for drawable objects and groups.
    """

    _model_name = Unicode('ObjectModel').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)
    _model_module_version = Unicode('~' + __version__).tag(sync=True)

    id = Int().tag(sync=True)

    def __init__(self, **kwargs):
        self.id = id(self)

        super(Drawable, self).__init__(**kwargs)

    def __iter__(self):
        return (self,).__iter__()

    def __add__(self, other):
        return Group(self, other)

    def fetch_data(self, field):
        self.send({'msg_type': 'fetch', 'field': field})

    def _ipython_display_(self, **kwargs):
        """Called when `IPython.display.display` is called on the widget."""
        from IPython.display import display
        import k3d
        plot = k3d.plot()
        plot += self
        return display(plot)


class Group(Drawable):
    """
    An aggregated group of Drawables, itself a Drawable.

    It can be inserted or removed from a Plot including all members.
    """
    __objs = None

    def __init__(self, *args):
        self.__objs = tuple(self.__assert_drawable(drawable) for drawables in args for drawable in drawables)

    def __iter__(self):
        return self.__objs.__iter__()

    @staticmethod
    def __assert_drawable(arg):
        assert isinstance(arg, Drawable)

        return arg


class Line(Drawable):
    """
    Drawable
    """

    type = Unicode(default_value='Line', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    line_width = Float().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
    point_positions = Array().tag(sync=True, **array_serialization)


class MarchingCubes(Drawable):
    type = Unicode(default_value='MarchingCubes', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    width = Int().tag(sync=True)
    height = Int().tag(sync=True)
    length = Int().tag(sync=True)
    level = Float().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
    scalar_field = Array().tag(sync=True, **array_serialization)


class Points(Drawable):
    type = Unicode(default_value='Points', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    shader = Unicode().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
    point_size = Float().tag(sync=True)
    point_colors = Array().tag(sync=True, **array_serialization)
    point_positions = Array().tag(sync=True, **array_serialization)

    @validate('point_colors')
    def _validate_colors(self, proposal):
        required = self.point_positions.size
        actual = proposal['value'].size
        if actual != 0 and required != actual:
            raise TraitError('point_colors has wrong size: %s (%s required)' % (actual, required))
        return proposal['value']


class STL(Drawable):
    type = Unicode(default_value='STL', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
    text = Unicode().tag(sync=True)
    binary = Array().tag(sync=True, **array_serialization)


class Surface(Drawable):
    type = Unicode(default_value='Surface', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    width = Int().tag(sync=True)
    height = Int().tag(sync=True)
    heights = Array().tag(sync=True, **array_serialization)
    model_matrix = Array().tag(sync=True, **array_serialization)


class Text(Drawable):
    type = Unicode(default_value='Text', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    size = Float().tag(sync=True)
    font_face = Unicode().tag(sync=True)
    font_weight = Int().tag(sync=True)
    font_size = Int().tag(sync=True)
    position = List().tag(sync=True)
    text = Unicode().tag(sync=True)


class Text2d(Drawable):
    type = Unicode(default_value='Text2d', read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    size = Float().tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    position = List().tag(sync=True)
    text = Unicode().tag(sync=True)


class Texture(Drawable):
    type = Unicode(default_value='Texture', read_only=True).tag(sync=True)
    binary = Bytes().tag(sync=True)
    file_format = Unicode().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)


class Vectors(Drawable):
    type = Unicode(default_value='Vectors', read_only=True).tag(sync=True)
    colors = Array().tag(sync=True, **array_serialization)
    head_color = Int().tag(sync=True)
    origin_color = Int().tag(sync=True)
    labels = List().tag(sync=True)
    head_size = Float().tag(sync=True)
    label_size = Float().tag(sync=True)
    line_width = Float().tag(sync=True)
    use_head = Bool().tag(sync=True)
    origins = Array().tag(sync=True, **array_serialization)
    vectors = Array().tag(sync=True, **array_serialization)
    model_matrix = Array().tag(sync=True, **array_serialization)


class VectorFields(Drawable):
    type = Unicode(default_value='VectorFields', read_only=True).tag(sync=True)
    colors = Array().tag(sync=True, **array_serialization)
    head_color = Int().tag(sync=True)
    width = Int().tag(sync=True)
    height = Int().tag(sync=True)
    length = Int(allow_none=True).tag(sync=True)
    origin_color = Int().tag(sync=True)
    use_head = Bool().tag(sync=True)
    head_size = Float().tag(sync=True)
    scale = Float().tag(sync=True)
    vectors = Array().tag(sync=True, **array_serialization)
    model_matrix = Array().tag(sync=True, **array_serialization)


class Voxels(Drawable):
    type = Unicode(default_value='Voxels', read_only=True).tag(sync=True)
    color_map = Array().tag(sync=True, **array_serialization)
    width = Int().tag(sync=True)
    height = Int().tag(sync=True)
    length = Int().tag(sync=True)
    voxels = Array().tag(sync=True, **array_serialization)
    model_matrix = Array().tag(sync=True, **array_serialization)


class Mesh(Drawable):
    type = Unicode(default_value='Mesh', read_only=True).tag(sync=True)
    vertices = Array().tag(sync=True, **array_serialization)
    indices = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization)
    color_map = Array().tag(sync=True, **array_serialization)
    color_range = List().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
