import ipywidgets as widgets
from traitlets import Unicode, Int, Float, List, Bool, Bytes, Integer
from traitlets import validate, TraitError
from traittypes import Array
from .helpers import array_serialization
from ._version import __version__
import numpy as np


class ListOrArray(List):
    _cast_types = (tuple, np.ndarray)

    def __init__(self, *args, **kwargs):
        self._empty_ok = kwargs.pop('empty_ok', False)
        super(ListOrArray, self).__init__(*args, **kwargs)

    def validate_elements(self, obj, value):
        if self._empty_ok and len(value) == 0:
            return list(value)
        return super(ListOrArray, self).validate_elements(obj, value)


class Drawable(widgets.Widget):
    """
    Base class for drawable objects and groups.
    """

    _model_name = Unicode('ObjectModel').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)
    _model_module_version = Unicode('~' + __version__).tag(sync=True)

    id = Integer().tag(sync=True)
    visible = Bool(True).tag(sync=True)
    compression_level = Integer().tag(sync=True)

    def __init__(self, **kwargs):
        self.id = id(self)

        super(Drawable, self).__init__(**kwargs)

    def __iter__(self):
        return (self,).__iter__()

    def __add__(self, other):
        return Group(self, other)

    def fetch_data(self, field):
        """Request updating the value of a field modified in browser.

        For data modified in the widget on the browser side, this triggers an asynchronous
        update of the value in the Python kernel.

        Only specific features require this mechanism, e.g. the in-browser editing of voxels.

        Arguments:
            field: `str`. the field name."""
        self.send({'msg_type': 'fetch', 'field': field})

    def push_data(self, field):
        """Request updating the value of a field modified in backend.

        For data modified in the backend side, this triggers an asynchronous
        update of the value in the browser widget.

        Only specific features require this mechanism, e.g. the in-browser editing of voxels.

        Arguments:
            field: `str`. the field name."""
        self.notify_change({'name': field, 'type': 'change'})

    def _ipython_display_(self, **kwargs):
        """Called when `IPython.display.display` is called on the widget."""
        import k3d
        plot = k3d.plot()
        plot += self
        plot.display()


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

    def __setattr__(self, key, value):
        """Special method override which allows for setting model matrix for all members of the group."""
        if key == 'model_matrix':
            for d in self:
                d.model_matrix = value
        else:
            super(Group, self).__setattr__(key, value)

    @staticmethod
    def __assert_drawable(arg):
        assert isinstance(arg, Drawable)

        return arg


# DRAWABLE OBJECTS


class Line(Drawable):
    """
    A path (polyline) made up of line segments.

    Attributes:
        vertices: `array_like`. An array with (x, y, z) coordinates of segment endpoints.
        colors: `array_like`. Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`. Packed RGB color of the lines (0xff0000 is red, 0xff is blue) when `colors` is empty.
        attribute: `array_like`. Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`. A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`. A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        width: `float`. The thickness of the lines.
        shader: `str`. Display style (name of the shader used) of the lines.
            Legal values are:
            `simple`: simple lines,
            `thick`: thick lines,
            `mesh`: high precision triangle mesh of segments (high quality and GPU load).
        radial_segments: 'int': Number of segmented faces around the circumference of the tube
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    vertices = Array().tag(sync=True, **array_serialization)
    colors = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    width = Float().tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization)
    color_map = Array().tag(sync=True, **array_serialization)
    color_range = ListOrArray(minlen=2, maxlen=2, empty_ok=True).tag(sync=True)
    shader = Unicode().tag(sync=True)
    radial_segments = Int().tag(sync=True)

    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Line, self).__init__(**kwargs)

        self.set_trait('type', 'Line')

    @validate('colors')
    def _validate_colors(self, proposal):
        required = self.vertices.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal['value'].size
        if actual != 0 and required != actual:
            raise TraitError('colors has wrong size: %s (%s required)' % (actual, required))
        return proposal['value']


class MarchingCubes(Drawable):
    """
    An isosurface in a scalar field obtained through Marching Cubes algorithm.

    The default domain of the scalar field is -0.5 < x, y, z < 0.5.
    If the domain should be different, the bounding box needs to be transformed using the model_matrix.

    Attributes:
        scalar_field: `array_like`. A 3D scalar field of values.
        level: `float`. Value at the computed isosurface.
        color: `int`. Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue).
        wireframe: `bool`. Whether mesh should display as wireframe.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    scalar_field = Array().tag(sync=True, **array_serialization)
    level = Float().tag(sync=True)
    color = Int().tag(sync=True)
    wireframe = Bool().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(MarchingCubes, self).__init__(**kwargs)

        self.set_trait('type', 'MarchingCubes')


class Mesh(Drawable):
    """
    A 3D triangles mesh.

    Attributes:
        vertices: `array_like`. Array of triangle vertices: float (x, y, z) coordinate triplets.
        indices: `array_like`.  Array of vertex indices: int triplets of indices from vertices array.
        color: `int`. Packed RGB color of the mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        attribute: `array_like`. Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`. A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`. A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        wireframe: `bool`. Whether mesh should display as wireframe.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    vertices = Array().tag(sync=True, **array_serialization)
    indices = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization)
    color_map = Array().tag(sync=True, **array_serialization)
    color_range = ListOrArray(minlen=2, maxlen=2, empty_ok=True).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Mesh, self).__init__(**kwargs)

        self.set_trait('type', 'Mesh')


class Points(Drawable):
    """
    A point cloud.

    Attributes:
        positions: `array_like`. Array with (x, y, z) coordinates of the points.
        colors: `array_like`. Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`. Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        point_size: `float`. Diameter of the balls representing the points in 3D space.
        shader: `str`. Display style (name of the shader used) of the points.
            Legal values are:
            `flat`: simple circles with uniform color,
            `3d`: little 3D balls,
            `3dSpecular`: little 3D balls with specular lightning,
            `mesh`: high precision triangle mesh of a ball (high quality and GPU load).
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    positions = Array().tag(sync=True, **array_serialization)
    colors = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    point_size = Float().tag(sync=True)
    shader = Unicode().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Points, self).__init__(**kwargs)

        self.set_trait('type', 'Points')

    @validate('colors')
    def _validate_colors(self, proposal):
        required = self.positions.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal['value'].size
        if actual != 0 and required != actual:
            raise TraitError('colors has wrong size: %s (%s required)' % (actual, required))
        return proposal['value']


class STL(Drawable):
    """
    A STereoLitograpy 3D geometry.

    STL is a popular format introduced for 3D printing. There are two sub-formats - ASCII and binary.

    Attributes:
        text: `str`. STL data in text format (ASCII STL).
        binary: `bytes`. STL data in binary format (Binary STL).
            The `text` attribute should be set to None when using Binary STL.
        color: `int`. Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`. 4x4 model transform matrix.
        wireframe: `bool`. Whether mesh should display as wireframe.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = Unicode(allow_none=True).tag(sync=True)
    binary = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)
    wireframe = Bool().tag(sync=True)

    def __init__(self, **kwargs):
        super(STL, self).__init__(**kwargs)

        self.set_trait('type', 'STL')


class Surface(Drawable):
    """
    Surface plot of a 2D function z = f(x, y).

    The default domain of the scalar field is -0.5 < x, y < 0.5.
    If the domain should be different, the bounding box needs to be transformed using the model_matrix.

    Attributes:
        heights: `array_like`. 2D scalar field of Z values.
        color: `int`. Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        wireframe: `bool`. Whether mesh should display as wireframe.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    heights = Array().tag(sync=True, **array_serialization)
    color = Int().tag(sync=True)
    wireframe = Bool().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Surface, self).__init__(**kwargs)

        self.set_trait('type', 'Surface')


class Text(Drawable):
    """
    Text rendered using KaTeX with a 3D position.

    Attributes:
        text: `str`. Content of the text.
        position: `list`. Coordinates (x, y, z) of the text's position.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`. Two-letter string representing the text's alignment.
            First letter: 'l', 'c' or 'r': left, center or right
            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`. Font size in 'em' HTML units.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = Unicode().tag(sync=True)
    position = ListOrArray(minlen=3, maxlen=3).tag(sync=True)
    color = Int().tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    size = Float().tag(sync=True)

    def __init__(self, **kwargs):
        super(Text, self).__init__(**kwargs)

        self.set_trait('type', 'Text')


class Text2d(Drawable):
    """
    Text rendered using KaTeX with a fixed 2D position, independent of camera settings.

    Attributes:
        text: `str`. Content of the text.
        position: `list`. Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        reference_point: `str`. Two-letter string representing the text's alignment.
            First letter: 'l', 'c' or 'r': left, center or right
            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`. Font size in 'em' HTML units.
    """

    type = Unicode(read_only=True).tag(sync=True)
    color = Int().tag(sync=True)
    size = Float().tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    position = ListOrArray(minlen=2, maxlen=2).tag(sync=True)
    text = Unicode().tag(sync=True)

    def __init__(self, **kwargs):
        super(Text2d, self).__init__(**kwargs)

        self.set_trait('type', 'Text2d')


class Texture(Drawable):
    """
    A 2D image displayed as a texture.

    By default, the texture image is mapped into the square: -0.5 < x, y < 0.5, z = 1.
    If the size (scale, aspect ratio) or position should be different then the texture should be transformed
    using the model_matrix.

    Attributes:
        binary: `bytes`. Image data in a specific format.
        file_format: `str`. Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        attribute: `array_like`. Array of float attribute for the color mapping, coresponding to each pixels.
        color_map: `list`. A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`. A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    binary = Bytes(allow_none=True).tag(sync=True)
    file_format = Unicode(allow_none=True).tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization)
    color_map = Array().tag(sync=True, **array_serialization)
    color_range = ListOrArray(minlen=2, maxlen=2, empty_ok=True).tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Texture, self).__init__(**kwargs)

        self.set_trait('type', 'Texture')


class TextureText(Drawable):
    """
    A text in the 3D space rendered using a texture.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Attributes:
        text: `str`. Content of the text.
        position: `list`. Coordinates (x, y, z) of the text's position.
        color: `int`. Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        size: `float`. Size of the texture sprite containing the text.
        font_face: `str`. Name of the font to use for rendering the text.
        font_weight: `int`. Thickness of the characters in HTML-like units from the range (100, 900), where
            400 is normal and 600 is bold font.
        font_size: `int`. The font size inside the sprite texture in px units. This does not affect the size of the
            text in the scene, only the accuracy and raster size of the texture.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = Unicode().tag(sync=True)
    position = ListOrArray(minlen=3, maxlen=3).tag(sync=True)
    color = Int().tag(sync=True)
    size = Float().tag(sync=True)
    font_face = Unicode().tag(sync=True)
    font_weight = Int().tag(sync=True)
    font_size = Int().tag(sync=True)

    def __init__(self, **kwargs):
        super(TextureText, self).__init__(**kwargs)

        self.set_trait('type', 'TextureText')


class VectorField(Drawable):
    """
    A dense 3D or 2D vector field.

    By default, the origins of the vectors are assumed to be a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    or -0.5 < x, y < 0.5 square, regardless of the passed vector field shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For sparse (i.e. not forming a grid) 3D vectors, use the `Vectors` drawable.

    Attributes:
        vectors: `array_like`. Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
        colors: `array_like`. Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`. Packed RGB color of the origins (0xff0000 is red, 0xff is blue) when `colors` is empty.
        head_color: `int`. Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue) when `colors` is empty.
        use_head: `bool`. Whether vectors should display an arrow head.
        head_size: `float`. The size of the arrow heads.
        scale: `float`. Scale factor for the vector lengths, for artificially scaling the vectors in place.
        line_width: `float`. Width of the vector segments.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    vectors = Array().tag(sync=True, **array_serialization)
    colors = Array().tag(sync=True, **array_serialization)
    origin_color = Int().tag(sync=True)
    head_color = Int().tag(sync=True)
    use_head = Bool().tag(sync=True)
    head_size = Float().tag(sync=True)
    scale = Float().tag(sync=True)
    line_width = Float().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(VectorField, self).__init__(**kwargs)

        self.set_trait('type', 'VectorField')

    @validate('vectors')
    def _validate_vectors(self, proposal):
        shape = proposal['value'].shape
        if len(shape) not in (3, 4) or len(shape) != shape[-1] + 1:
            raise TraitError('Vector field has invalid shape: {}, '
                             'expected (L, H, W, 3) for a 3D or (H, W, 2) for a 2D field'.format(shape))
        return np.array(proposal['value'], np.float32)


class Vectors(Drawable):
    """
    3D vectors.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For dense (i.e. forming a grid) 3D or 2D vectors, use the `VectorField` drawable.

    Attributes:
        vectors: `array_like`. The vectors as (dx, dy, dz) float triples.
        origins: `array_like`. Same-size array of (x, y, z) coordinates of vector origins.
        colors: `array_like`. Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`. Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`. Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        use_head: `bool`. Whether vectors should display an arrow head.
        head_size: `float`. The size of the arrow heads.
        labels: `list` of `str`. Captions to display next to the vectors.
        label_size: `float`. Label font size in 'em' HTML units.
        line_width: `float`. Width of the vector segments.
        model_matrix: `array_like`. 4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    origins = Array().tag(sync=True, **array_serialization)
    vectors = Array().tag(sync=True, **array_serialization)
    colors = Array().tag(sync=True, **array_serialization)
    origin_color = Int().tag(sync=True)
    head_color = Int().tag(sync=True)
    use_head = Bool().tag(sync=True)
    head_size = Float().tag(sync=True)
    labels = List().tag(sync=True)
    label_size = Float().tag(sync=True)
    line_width = Float().tag(sync=True)
    model_matrix = Array().tag(sync=True, **array_serialization)

    def __init__(self, **kwargs):
        super(Vectors, self).__init__(**kwargs)

        self.set_trait('type', 'Vectors')


class Voxels(Drawable):
    """
    3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    Attributes:
        voxels: `array_like`. 3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`. Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
            The color defined at index i is for voxel value (i+1), e.g.:
            color_map = [0xff, 0x00ff]
            voxels = [[[
                0, # empty voxel
                1, # blue voxel
                2  # red voxel
            ]]]
        model_matrix: `array_like`. 4x4 model transform matrix.
        wireframe: `bool`. Whether mesh should display as wireframe.
        outlines: `bool`. Whether mesh should display with outlines.
        outlines_color: `int`. Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)
    voxels = Array().tag(sync=True, **array_serialization)
    color_map = Array().tag(sync=True, **array_serialization)
    model_matrix = Array().tag(sync=True, **array_serialization)
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int().tag(sync=True)
    click_callback = None

    def __init__(self, **kwargs):
        super(Voxels, self).__init__(**kwargs)

        self.set_trait('type', 'Voxels')
        self.on_msg(self._handle_custom_msg)
        pass

    def _handle_custom_msg(self, content, buffers):
        if content.get('msg_type', '') == 'click_callback':
            if self.click_callback is not None:
                self.click_callback(content['coord']['x'], content['coord']['y'], content['coord']['z'])
