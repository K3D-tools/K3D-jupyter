import warnings

import ipywidgets as widgets
import numpy as np
from traitlets import (
    Any,
    Bool,
    Bytes,
    Dict,
    Float,
    Int,
    Integer,
    List,
    TraitError,
    Unicode,
    Union,
    validate,
)
from traittypes import Array

from ._version import __version__ as version
from .helpers import (
    array_serialization_wrap,
    callback_serialization_wrap,
    get_bounding_box_point,
    get_bounding_box_points,
    get_bounding_box,
    shape_validation,
    validate_sparse_voxels,
)
from .validation.stl import AsciiStlData, BinaryStlData

EPSILON = np.finfo(np.float32).eps


class TimeSeries(Union):
    def __init__(self, trait):
        if isinstance(trait, list):
            Union.__init__(self, trait + [Dict(t) for t in trait])
        else:
            Union.__init__(self, [trait, Dict(trait)])


class ListOrArray(List):
    _cast_types = (tuple, np.ndarray)

    def __init__(self, *args, **kwargs):
        self._empty_ok = kwargs.pop("empty_ok", False)
        List.__init__(self, *args, **kwargs)

    def validate_elements(self, obj, value):
        if self._empty_ok and len(value) == 0:
            return list(value)
        return super(ListOrArray, self).validate_elements(obj, value)


class VoxelChunk(widgets.Widget):
    """
    Voxel chunk class for selective updating voxels
    """

    _model_name = Unicode("ChunkModel").tag(sync=True)
    _model_module = Unicode("k3d").tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)

    id = Int().tag(sync=True)
    voxels = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap("voxels"))
    coord = Array(dtype=np.uint32).tag(sync=True, **array_serialization_wrap("coord"))
    multiple = Int().tag(sync=True)
    compression_level = Integer().tag(sync=True)

    def push_data(self, field):
        self.notify_change({"name": field, "type": "change"})

    def __init__(self, **kwargs):
        self.id = id(self)
        super(VoxelChunk, self).__init__(**kwargs)

    def __getitem__(self, name):
        return getattr(self, name)


class Drawable(widgets.Widget):
    """
    Base class for drawable objects and groups.
    """

    _model_name = Unicode("ObjectModel").tag(sync=True)
    _model_module = Unicode("k3d").tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)

    id = Integer().tag(sync=True)
    name = Unicode(default_value=None, allow_none=True).tag(sync=True)
    group = Unicode(default_value=None, allow_none=True).tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)
    visible = TimeSeries(Bool(True)).tag(sync=True)
    compression_level = Integer().tag(sync=True)

    def __getitem__(self, name):
        return getattr(self, name)

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
            field: `str`.
                The field name."""
        self.send({"msg_type": "fetch", "field": field})

    def push_data(self, field):
        """Request updating the value of a field modified in backend.

        For data modified in the backend side, this triggers an asynchronous
        update of the value in the browser widget.

        Only specific features require this mechanism, e.g. the in-browser editing of voxels.

        Arguments:
            field: `str`.
                The field name."""
        self.notify_change({"name": field, "type": "change"})

    def _ipython_display_(self, **kwargs):
        """Called when `IPython.display.display` is called on the widget."""
        import k3d

        plot = k3d.plot()
        plot += self
        plot.display()

    def clone(self):
        return clone_object(self)


class DrawableWithVoxelCallback(Drawable):
    """
    Base class for drawable with voxels callback handling
    """

    click_callback = None
    hover_callback = None

    def __init__(self, **kwargs):
        super(DrawableWithVoxelCallback, self).__init__(**kwargs)

        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content, buffers):
        if content.get("msg_type", "") == "click_callback":
            if self.click_callback is not None:
                self.click_callback(
                    content["coord"]["x"], content["coord"]["y"], content["coord"]["z"]
                )

        if content.get("msg_type", "") == "hover_callback":
            if self.hover_callback is not None:
                self.hover_callback(
                    content["coord"]["x"], content["coord"]["y"], content["coord"]["z"]
                )


class DrawableWithCallback(Drawable):
    """
    Base class for drawable with callback handling
    """

    click_callback = Any(default_value=None, allow_none=True).tag(
        sync=True, **callback_serialization_wrap("click_callback")
    )
    hover_callback = Any(default_value=None, allow_none=True).tag(
        sync=True, **callback_serialization_wrap("hover_callback")
    )

    def __init__(self, **kwargs):
        super(DrawableWithCallback, self).__init__(**kwargs)

        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content, buffers):
        if content.get("msg_type", "") == "click_callback":
            if self.click_callback is not None:
                self.click_callback(content)

        if content.get("msg_type", "") == "hover_callback":
            if self.hover_callback is not None:
                self.hover_callback(content)


class Group(Drawable):
    """
    An aggregated group of Drawables, itself a Drawable.

    It can be inserted or removed from a Plot including all members.
    """

    __objs = None

    def __init__(self, *args):
        self.__objs = tuple(
            self.__assert_drawable(drawable)
            for drawables in args
            for drawable in drawables
        )

    def __iter__(self):
        return self.__objs.__iter__()

    def __setattr__(self, key, value):
        """Special method override which allows for setting model matrix for all members of the group."""
        if key == "model_matrix":
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
        vertices: `array_like`.
            An array with (x, y, z) coordinates of segment endpoints.
        colors: `array_like`.
            Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`.
            Packed RGB color of the lines (0xff0000 is red, 0xff is blue) when `colors` is empty.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        width: `float`.
            The thickness of the lines.
        opacity: `float`.
            Opacity of lines.
        shader: `str`.
            Display style (name of the shader used) of the lines.
            Legal values are:

            :`simple`: simple lines,

            :`thick`: thick lines,

            :`mesh`: high precision triangle mesh of segments (high quality and GPU load).
        radial_segments: 'int':
            Number of segmented faces around the circumference of the tube.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)

    vertices = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("vertices")
    )
    colors = TimeSeries(Array(dtype=np.uint32)).tag(
        sync=True, **array_serialization_wrap("colors")
    )
    color = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    width = TimeSeries(Float(min=EPSILON, default_value=0.01)).tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    shader = TimeSeries(Unicode()).tag(sync=True)
    radial_segments = TimeSeries(Int()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Line, self).__init__(**kwargs)

        self.set_trait("type", "Line")

    def get_bounding_box(self):
        return get_bounding_box_points(self.vertices, self.model_matrix)

    @validate("colors")
    def _validate_colors(self, proposal):
        if type(proposal["value"]) is dict or type(self.vertices) is dict:
            return proposal["value"]

        required = self.vertices.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "colors has wrong size: %s (%s required)" % (actual, required)
            )
        return proposal["value"]


class Lines(Drawable):
    """
    A set of line (polyline) made up of indices.

    Attributes:
        vertices: `array_like`.
            An array with (x, y, z) coordinates of segment endpoints.
        indices: `array_like`.
            Array of vertex indices: int pair of indices from vertices array.
       indices_type: `str`.
            Interpretation of indices array
            Legal values are:

            :`segment`: indices contains pair of values,

            :`triangle`: indices contains triple of values
        colors: `array_like`.
            Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`.
            Packed RGB color of the lines (0xff0000 is red, 0xff is blue) when `colors` is empty.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        width: `float`.
            The thickness of the lines.
        opacity: `float`.
            Opacity of lines.
        shader: `str`.
            Display style (name of the shader used) of the lines.
            Legal values are:

            :`simple`: simple lines,

            :`thick`: thick lines,

            :`mesh`: high precision triangle mesh of segments (high quality and GPU load).
        radial_segments: 'int':
            Number of segmented faces around the circumference of the tube.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)

    vertices = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("vertices")
    )
    indices = Array(dtype=np.float32).tag(sync=True, **array_serialization_wrap("indices"))
    indices_type = TimeSeries(Unicode()).tag(sync=True)
    colors = TimeSeries(Array(dtype=np.uint32)).tag(
        sync=True, **array_serialization_wrap("colors")
    )
    color = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    width = TimeSeries(Float(min=EPSILON, default_value=0.01)).tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    shader = TimeSeries(Unicode()).tag(sync=True)
    radial_segments = TimeSeries(Int()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Lines, self).__init__(**kwargs)

        self.set_trait("type", "Lines")

    def get_bounding_box(self):
        return get_bounding_box_points(self.vertices, self.model_matrix)

    @validate("colors")
    def _validate_colors(self, proposal):
        if type(proposal["value"]) is dict or type(self.vertices) is dict:
            return proposal["value"]

        required = self.vertices.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "colors has wrong size: %s (%s required)" % (actual, required)
            )
        return proposal["value"]


class MarchingCubes(DrawableWithCallback):
    """
    An isosurface in a scalar field obtained through Marching Cubes algorithm.

    The default domain of the scalar field is -0.5 < x, y, z < 0.5.
    If the domain should be different, the bounding box needs to be transformed using the model_matrix.

    Attributes:
        scalar_field: `array_like`.
            A 3D scalar field of values.
        level: `float`.
            Value at the computed isosurface.
        spacings_x: `array_like`.
            A spacings in x axis. Should match to scalar_field shape.
        spacings_y: `array_like`.
            A spacings in y axis. Should match to scalar_field shape.
        spacings_z: `array_like`.
            A spacings in z axis. Should match to scalar_field shape.
        color: `int`.
            Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        opacity: `float`.
            Opacity of mesh.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    scalar_field = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("scalar_field")
    )
    spacings_x = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_x")
    )
    spacings_y = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_y")
    )
    spacings_z = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("spacings_z")
    )
    level = Float().tag(sync=True)
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)

    def __init__(self, **kwargs):
        super(MarchingCubes, self).__init__(**kwargs)

        self.set_trait("type", "MarchingCubes")


class Mesh(DrawableWithCallback):
    """
    A 3D triangles mesh.

    Attributes:
        vertices: `array_like`.
            Array of triangle vertices: float (x, y, z) coordinate triplets.
        indices: `array_like`.
            Array of vertex indices: int triplets of indices from vertices array.
        color: `int`.
            Packed RGB color of the mesh (0xff0000 is red, 0xff is blue) when not using color maps.
        colors: `array_like`.
            Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        triangles_attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each triangle.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        opacity: `float`.
            Opacity of mesh.
        volume: `array_like`.
            3D array of `float`
        volume_bounds: `array_like`.
            6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1)
        texture: `bytes`.
            Image data in a specific format.
        texture_file_format: `str`.
            Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        uvs: `array_like`.
            Array of float uvs for the texturing, coresponding to each vertex.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    vertices = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("vertices")
    )
    indices = TimeSeries(Array(dtype=np.uint32)).tag(
        sync=True, **array_serialization_wrap("indices")
    )
    color = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    colors = TimeSeries(Array(dtype=np.uint32)).tag(
        sync=True, **array_serialization_wrap("colors")
    )
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    triangles_attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("triangles_attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    wireframe = TimeSeries(Bool()).tag(sync=True)
    flat_shading = TimeSeries(Bool()).tag(sync=True)
    side = TimeSeries(Unicode()).tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    volume = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("volume"))
    volume_bounds = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("volume_bounds")
    )
    texture = Bytes(allow_none=True).tag(sync=True, **array_serialization_wrap("texture"))
    texture_file_format = Unicode(allow_none=True).tag(sync=True)
    uvs = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("uvs"))
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Mesh, self).__init__(**kwargs)

        self.set_trait("type", "Mesh")

    @validate("colors")
    def _validate_colors(self, proposal):
        if type(proposal["value"]) is dict or type(self.vertices) is dict:
            return proposal["value"]

        required = self.vertices.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "colors has wrong size: %s (%s required)" % (actual, required)
            )
        return proposal["value"]

    @validate("volume")
    def _validate_volume(self, proposal):
        if type(proposal["value"]) is dict:
            return proposal["value"]

        if type(proposal["value"]) is np.ndarray and proposal[
            "value"
        ].dtype is np.dtype(object):
            return proposal["value"].tolist()

        if proposal["value"].shape == (0,):
            return np.array(proposal["value"], dtype=np.float32)

        required = [np.float16, np.float32]
        actual = proposal["value"].dtype

        if actual not in required:
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required))

            return proposal["value"].astype(np.float32)

        return proposal["value"]

    def get_bounding_box(self):
        return get_bounding_box_points(self.vertices, self.model_matrix)


class Points(Drawable):
    """
    A point cloud.

    Attributes:
        positions: `array_like`.
            Array with (x, y, z) coordinates of the points.
        colors: `array_like`.
            Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        color: `int`.
            Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        point_size: `float`.
            Diameter of the balls representing the points in 3D space.
        point_sizes: `array_like`.
            Same-length array of `float` sizes of the points.
        shader: `str`.
            Display style (name of the shader used) of the points.
            Legal values are:

            :`flat`: simple circles with uniform color,

            :`dot`: simple dot with uniform color,

            :`3d`: little 3D balls,

            :`3dSpecular`: little 3D balls with specular lightning,

            :`mesh`: high precision triangle mesh of a ball (high quality and GPU load).
        mesh_detail: `int`.
            Default is 2. Setting this to a value greater than 0 adds more vertices making it no longer an
            icosahedron. When detail is greater than 1, it's effectively a sphere. Only valid if shader='mesh'
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each point.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    positions = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("positions")
    )
    colors = TimeSeries(Array(dtype=np.uint32)).tag(
        sync=True, **array_serialization_wrap("colors")
    )
    color = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    point_size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
    point_sizes = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("point_sizes")
    )
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    opacities = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacities")
    )
    shader = TimeSeries(Unicode()).tag(sync=True)
    mesh_detail = TimeSeries(Int(min=0, max=12)).tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Points, self).__init__(**kwargs)

        self.set_trait("type", "Points")

    @validate("colors")
    def _validate_colors(self, proposal):
        if type(proposal["value"]) is dict or type(self.positions) is dict:
            return proposal["value"]

        required = self.positions.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "colors has wrong size: %s (%s required)" % (actual, required)
            )
        return proposal["value"]

    def get_bounding_box(self):
        return get_bounding_box_points(self.positions, self.model_matrix)


class STL(Drawable):
    """
    A STereoLitograpy 3D geometry.

    STL is a popular format introduced for 3D printing. There are two sub-formats - ASCII and binary.

    Attributes:
        text: `str`.
            STL data in text format (ASCII STL).
        binary: `bytes`.
            STL data in binary format (Binary STL).
            The `text` attribute should be set to None when using Binary STL.
        color: `int`.
            Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = AsciiStlData(allow_none=True, default_value=None).tag(sync=True)
    binary = BinaryStlData(allow_none=True,
                           default_value=None).tag(sync=True, **array_serialization_wrap("binary"))
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(STL, self).__init__(**kwargs)

        self.set_trait("type", "STL")

    def get_bounding_box(self):
        warnings.warn("STL bounding box is still not supported")
        return [-1, 1, -1, 1, -1, 1]


class Surface(DrawableWithCallback):
    """
    Surface plot of a 2D function z = f(x, y).

    The default domain of the scalar field is -0.5 < x, y < 0.5.
    If the domain should be different, the bounding box needs to be transformed using the model_matrix.

    Attributes:
        heights: `array_like`.
            2D scalar field of Z values.
        color: `int`.
            Packed RGB color of the resulting mesh (0xff0000 is red, 0xff is blue).
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        flat_shading: `bool`.
            Whether mesh should display with flat shading.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    heights = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("heights")
    )
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Surface, self).__init__(**kwargs)

        self.set_trait("type", "Surface")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class Text(Drawable):
    """
    Text rendered using KaTeX with a 3D position.

    Attributes:
        text: `str`.
            Content of the text.
        position: `list`.
            Coordinates (x, y, z) of the text's position.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        is_html: `Boolean`.
            Whether text should be interpreted as HTML insted of KaTeX.
        on_top: `Boolean`.
            Render order with 3d object
        reference_point: `str`.
            Two-letter string representing the text's alignment.

            First letter: 'l', 'c' or 'r': left, center or right

            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`.
            Font size in 'em' HTML units.
        label_box: `Boolean`.
            Label background box.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = TimeSeries(Unicode()).tag(sync=True)
    position = TimeSeries(ListOrArray(minlen=3, maxlen=3)).tag(sync=True)
    is_html = Bool(False).tag(sync=True)
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
    on_top = Bool().tag(sync=True)
    label_box = Bool().tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Text, self).__init__(**kwargs)

        self.set_trait("type", "Text")

    def get_bounding_box(self):
        return get_bounding_box_point(self.position)


class Text2d(Drawable):
    """
    Text rendered using KaTeX with a fixed 2D position, independent of camera settings.

    Attributes:
        text: `str`.
            Content of the text.
        position: `list`.
            Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        is_html: `Boolean`.
            Whether text should be interpreted as HTML insted of KaTeX.
        reference_point: `str`.
            Two-letter string representing the text's alignment.

            First letter: 'l', 'c' or 'r': left, center or right

            Second letter: 't', 'c' or 'b': top, center or bottom.
        size: `float`.
            Font size in 'em' HTML units.
        label_box: `Boolean`.
            Label background box.
    """

    type = Unicode(read_only=True).tag(sync=True)
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
    is_html = Bool(False).tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    position = TimeSeries(ListOrArray(minlen=2, maxlen=2)).tag(sync=True)
    text = TimeSeries(Unicode()).tag(sync=True)
    label_box = Bool().tag(sync=True)

    def __init__(self, **kwargs):
        super(Text2d, self).__init__(**kwargs)

        self.set_trait("type", "Text2d")

    def get_bounding_box(self):
        return get_bounding_box_point(self.position)


class Label(Drawable):
    """
    Label rendered using KaTeX with a 3D position.

    Attributes:
        text: `str`.
            Content of the text.
        position: `list`.
            Coordinates (x, y, z) of the text's position.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        on_top: `Boolean`.
            Render order with 3d object
        label_box: `Boolean`.
            Label background box.
        mode: `str`.
            Label node. Can be 'dynamic', 'local' or 'side'.
        is_html: `Boolean`.
            Whether text should be interpreted as HTML insted of KaTeX.
        max_length: `float`.
            Maximum length of line in % of half screen size.
        size: `float`.
            Font size in 'em' HTML units.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    mode = Unicode().tag(sync=True)
    text = TimeSeries(Unicode()).tag(sync=True)
    is_html = Bool(False).tag(sync=True)
    position = TimeSeries(ListOrArray(minlen=3, maxlen=3)).tag(sync=True)
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    max_length = Float(min=0, max=1.0).tag(sync=True)
    size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
    on_top = Bool().tag(sync=True)
    label_box = Bool().tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Label, self).__init__(**kwargs)

        self.set_trait("type", "Label")

    def get_bounding_box(self):
        return get_bounding_box_point(self.position)


class Texture(DrawableWithCallback):
    """
    A 2D image displayed as a texture.

    By default, the texture image is mapped into the square: -0.5 < x, y < 0.5, z = 1.
    If the size (scale, aspect ratio) or position should be different then the texture should be transformed
    using the model_matrix.

    Attributes:
        binary: `bytes`.
            Image data in a specific format.
        file_format: `str`.
            Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each pixels.
        color_map: `list`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            tuples should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        interpolation: `bool`.
            Whether data should be interpolatedor not.
        puv: `list`.
            A list of float triplets (x,y,z). The first triplet mean a position of left-bottom corner of texture.
            Second and third triplets means a base of coordinate system for texture.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    binary = Bytes(allow_none=True).tag(sync=True, **array_serialization_wrap("binary"))
    file_format = Unicode(allow_none=True).tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization_wrap("attribute"))
    puv = Array(dtype=np.float32).tag(sync=True, **array_serialization_wrap("puv"))
    color_map = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    color_range = ListOrArray(minlen=2, maxlen=2, empty_ok=True).tag(sync=True)
    interpolation = TimeSeries(Bool()).tag(sync=True)
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Texture, self).__init__(**kwargs)

        self.set_trait("type", "Texture")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class TextureText(Drawable):
    """
    A text in the 3D space rendered using a texture.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Attributes:
        text: `str`.
            Content of the text.
        position: `list`.
            Coordinates (x, y, z) of the text's position.
        color: `int`.
            Packed RGB color of the text (0xff0000 is red, 0xff is blue).
        size: `float`.
            Size of the texture sprite containing the text.
        font_face: `str`.
            Name of the font to use for rendering the text.
        font_weight: `int`.
            Thickness of the characters in HTML-like units from the range (100, 900), where
            400 is normal and 600 is bold font.
        font_size: `int`.
            The font size inside the sprite texture in px units. This does not affect the size of the
            text in the scene, only the accuracy and raster size of the texture.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = TimeSeries(Unicode()).tag(sync=True)
    position = TimeSeries(ListOrArray(minlen=3, maxlen=3)).tag(sync=True)
    color = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
    font_face = Unicode().tag(sync=True)
    font_weight = Int().tag(sync=True)
    font_size = Int().tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(TextureText, self).__init__(**kwargs)

        self.set_trait("type", "TextureText")

    def get_bounding_box(self):
        return get_bounding_box_point(self.position)


class VectorField(Drawable):
    """
    A dense 3D or 2D vector field.

    By default, the origins of the vectors are assumed to be a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    or -0.5 < x, y < 0.5 square, regardless of the passed vector field shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For sparse (i.e. not forming a grid) 3D vectors, use the `Vectors` drawable.

    Attributes:
        vectors: `array_like`.
            Vector field of shape (L, H, W, 3) for 3D fields or (H, W, 2) for 2D fields.
        colors: `array_like`.
            Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`.
            Packed RGB color of the origins (0xff0000 is red, 0xff is blue) when `colors` is empty.
        head_color: `int`.
            Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue) when `colors` is empty.
        use_head: `bool`.
            Whether vectors should display an arrow head.
        head_size: `float`.
            The size of the arrow heads.
        scale: `float`.
            Scale factor for the vector lengths, for artificially scaling the vectors in place.
        line_width: `float`.
            Width of the vector segments.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    vectors = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("vectors")
    )
    colors = Array(dtype=np.uint32).tag(sync=True, **array_serialization_wrap("colors"))
    origin_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    head_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    use_head = Bool().tag(sync=True)
    head_size = Float(min=EPSILON, default_value=1.0).tag(sync=True)
    scale = Float().tag(sync=True)
    line_width = Float(min=EPSILON, default_value=0.01).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(VectorField, self).__init__(**kwargs)

        self.set_trait("type", "VectorField")

    @validate("vectors")
    def _validate_vectors(self, proposal):
        shape = proposal["value"].shape
        if len(shape) not in (3, 4) or len(shape) != shape[-1] + 1:
            raise TraitError(
                "Vector field has invalid shape: {}, "
                "expected (L, H, W, 3) for a 3D or (H, W, 2) for a 2D field".format(
                    shape
                )
            )
        return np.array(proposal["value"], np.float32)

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class Vectors(Drawable):
    """
    3D vectors.

    The color of the vectors is a gradient from origin_color to head_color. Heads, when used, have uniform head_color.

    For dense (i.e. forming a grid) 3D or 2D vectors, use the `VectorField` drawable.

    Attributes:
        vectors: `array_like`.
            The vectors as (dx, dy, dz) float triples.
        origins: `array_like`.
            Same-size array of (x, y, z) coordinates of vector origins.
        colors: `array_like`.
            Twice the length of vectors array of int: packed RGB colors
            (0xff0000 is red, 0xff is blue).
            The array has consecutive pairs (origin_color, head_color) for vectors in row-major order.
        origin_color: `int`.
            Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        head_color: `int`.
            Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as color.
        use_head: `bool`.
            Whether vectors should display an arrow head.
        head_size: `float`.
            The size of the arrow heads.
        labels: `list` of `str`.
            Captions to display next to the vectors.
        label_size: `float`.
            Label font size in 'em' HTML units.
        line_width: `float`.
            Width of the vector segments.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    origins = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("origins")
    )
    vectors = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("vectors")
    )
    colors = Array(dtype=np.uint32).tag(sync=True, **array_serialization_wrap("colors"))
    origin_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    head_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    use_head = Bool().tag(sync=True)
    head_size = Float(min=EPSILON, default_value=1.0).tag(sync=True)
    labels = List().tag(sync=True)
    label_size = Float(min=EPSILON, default_value=1.0).tag(sync=True)
    line_width = Float(min=EPSILON, default_value=0.01).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Vectors, self).__init__(**kwargs)

        self.set_trait("type", "Vectors")

    def get_bounding_box(self):
        return get_bounding_box_points(
            np.stack([self.origins, self.vectors]), self.model_matrix
        )


class Volume(Drawable):
    """
    3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).

    Attributes:
        volume: `array_like`.
            3D array of `float`.
        color_map: `array_like`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        samples: `float`.
            Number of iteration per 1 unit of space.
        alpha_coef: `float`.
            Alpha multiplier.
        shadow: `str`.
            Type of shadow on volume.

            Legal values are:

            :`off`: shadow disabled,

            :`on_demand`: update shadow map on demand ( self.shadow_map_update() ),

            :`dynamic`: update shadow map automaticaly every shadow_delay.
        shadow_delay: `float`.
            Minimum number of miliseconds between shadow map updates.
        shadow_res: `int`.
            Resolution of shadow map.
        interpolation: `bool`.
            Whether volume raycasting should interpolate data or not.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    volume = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("volume"))
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    samples = TimeSeries(Float()).tag(sync=True)
    alpha_coef = TimeSeries(Float()).tag(sync=True)
    gradient_step = TimeSeries(Float()).tag(sync=True)
    shadow = TimeSeries(Unicode()).tag(sync=True)
    shadow_res = TimeSeries(Int(min=31, max=513, default_value=128)).tag(sync=True)
    shadow_delay = TimeSeries(Float()).tag(sync=True)
    ray_samples_count = TimeSeries(Int(min=1, max=128, default_value=16)).tag(sync=True)
    focal_length = TimeSeries(Float()).tag(sync=True)
    focal_plane = TimeSeries(Float()).tag(sync=True)
    interpolation = TimeSeries(Bool()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Volume, self).__init__(**kwargs)

        self.set_trait("type", "Volume")

    @validate("volume")
    def _validate_volume(self, proposal):
        if type(proposal["value"]) is dict:
            return proposal["value"]

        if type(proposal["value"]) is np.ndarray and proposal[
            "value"
        ].dtype is np.dtype(object):
            return proposal["value"].tolist()

        required = [np.float16, np.float32]
        actual = proposal["value"].dtype

        if actual not in required:
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required))

            return proposal["value"].astype(np.float32)

        return proposal["value"]

    def shadow_map_update(self, direction=None):
        """Request updating the shadow map in browser."""

        self.send({"msg_type": "shadow_map_update", "direction": direction})

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class MIP(Drawable):
    """
    3D volumetric data.

    By default, the volume are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).

    Attributes:
        volume: `array_like`.
            3D array of `float`.
        color_map: `array_like`.
            A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
        opacity_function: `array`.
            A list of float tuples (attribute value, opacity), sorted by attribute value. The first
            typles should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0.
        color_range: `list`.
            A pair [min_value, max_value], which determines the levels of color attribute mapped
            to 0 and 1 in the color map respectively.
        samples: `float`.
            Number of iteration per 1 unit of space.
        gradient_step: `float`
            Gradient light step.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)
    volume = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("volume"))
    color_map = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    color_range = TimeSeries(ListOrArray(minlen=2, maxlen=2, empty_ok=True)).tag(
        sync=True
    )
    gradient_step = TimeSeries(Float()).tag(sync=True)
    samples = TimeSeries(Float()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(MIP, self).__init__(**kwargs)

        self.set_trait("type", "MIP")

    @validate("volume")
    def _validate_volume(self, proposal):
        if type(proposal["value"]) is dict:
            return proposal["value"]

        if type(proposal["value"]) is np.ndarray and proposal[
            "value"
        ].dtype is np.dtype(object):
            return proposal["value"].tolist()

        required = [np.float16, np.float32]
        actual = proposal["value"].dtype

        if actual not in required:
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required))

            return proposal["value"].astype(np.float32)

        return proposal["value"]

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class Voxels(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    Different grid size, shape and rotation can be obtained using model_matrix.

    Attributes:
        voxels: `array_like`.
            3D array of `int` in range (0, 255).
            0 means empty voxel, 1 and above refer to consecutive color_map entries.
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).

            The color defined at index i is for voxel value (i+1), e.g.:

           | color_map = [0xff, 0x00ff]
           | voxels =
           | [
           | 0, # empty voxel
           | 1, # blue voxel
           | 2  # red voxel
           | ]

        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)
    voxels = Array(dtype=np.uint8).tag(sync=True, **array_serialization_wrap("voxels"))
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("voxels")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(Voxels, self).__init__(**kwargs)

        self.set_trait("type", "Voxels")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class SparseVoxels(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    Attributes:
        sparse_voxels: `array_like`.
            2D array of `coords` in format [[x,y,z,v],[x,y,z,v]].
            v = 0 means empty voxel, 1 and above refer to consecutive color_map entries.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)
    sparse_voxels = (
        Array(dtype=np.uint16)
            .tag(sync=True, **array_serialization_wrap("sparse_voxels"))
            .valid(validate_sparse_voxels)
    )
    space_size = (
        Array(dtype=np.uint32)
            .tag(sync=True, **array_serialization_wrap("space_size"))
            .valid(shape_validation(3))
    )
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(SparseVoxels, self).__init__(**kwargs)

        self.set_trait("type", "SparseVoxels")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


class VoxelsGroup(DrawableWithVoxelCallback):
    """
    3D volumetric data.

    By default, the voxels are a grid inscribed in the -0.5 < x, y, z < 0.5 cube
    regardless of the passed voxel array shape (aspect ratio etc.).
    Different grid size, shape and rotation can be obtained using the model_matrix.

    Attributes:
        voxels_group: `array_like`.
            List of `chunks` in format {voxels: np.array, coord: [x,y,z], multiple: number}.
        space_size: `array_like`.
            Width, Height, Length of space
        color_map: `array_like`.
            Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue).
        model_matrix: `array_like`.
            4x4 model transform matrix.
        wireframe: `bool`.
            Whether mesh should display as wireframe.
        opacity: `float`.
            Opacity of voxels.
        outlines: `bool`.
            Whether mesh should display with outlines.
        outlines_color: `int`.
            Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue)
    """

    type = Unicode(read_only=True).tag(sync=True)

    _hold_remeshing = Bool(default_value=False).tag(sync=True)

    voxels_group = List().tag(sync=True, **array_serialization_wrap("voxels_group"))
    chunks_ids = List().tag(sync=True)

    space_size = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("space_size")
    )
    color_map = Array(dtype=np.uint32).tag(
        sync=True, **array_serialization_wrap("color_map")
    )
    wireframe = Bool().tag(sync=True)
    outlines = Bool().tag(sync=True)
    outlines_color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super(VoxelsGroup, self).__init__(**kwargs)

        self.set_trait("type", "VoxelsGroup")

    def get_bounding_box(self):
        return get_bounding_box(self.model_matrix)


objects_map = {
    'Line': Line,
    'Label': Label,
    'MIP': MIP,
    'MarchingCubes': MarchingCubes,
    'Mesh': Mesh,
    'Points': Points,
    'STL': STL,
    'SparseVoxels': SparseVoxels,
    'Surface': Surface,
    'Text': Text,
    'Text2d': Text2d,
    'Texture': Texture,
    'TextureText': TextureText,
    'VectorField': VectorField,
    'Vectors': Vectors,
    'Volume': Volume,
    'Voxels': Voxels,
    'VoxelsGroup': VoxelsGroup
}


def create_object(obj, is_chunk=False):
    from .helpers import from_json

    attributes = {
        k: from_json(obj[k]) for k in obj.keys() if k != 'type'
    }

    # force to use current version
    attributes['_model_module'] = 'k3d_pro'
    attributes['_model_module_version'] = version
    attributes['_view_module_version'] = version

    if is_chunk:
        return VoxelChunk(**attributes)
    else:
        return objects_map[obj['type']](**attributes)


def clone_object(obj):
    param = {}

    for k, v in obj.traits().items():
        if "sync" in v.metadata and k not in ['id', 'type']:
            param[k] = obj[k]

    return objects_map[obj['type']](**param)
