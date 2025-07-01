"""Geometric objects for K3D."""

import numpy as np
import warnings
from traitlets import Bool, Bytes, Float, Int, TraitError, Unicode, validate
from traittypes import Array

from .base import (EPSILON, Drawable, DrawableWithCallback, ListOrArray,
                   TimeSeries)
from ..helpers import array_serialization_wrap, get_bounding_box_points
from ..validation.stl import AsciiStlData, BinaryStlData


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
        shininess: `float`.
            Shininess of object material.
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
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
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
        shininess: `float`.
            Shininess of object material.
        radial_segments: 'int':
            Number of segmented faces around the circumference of the tube.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)

    vertices = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("vertices")
    )
    indices = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("indices")
    )
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
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
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


class Mesh(DrawableWithCallback):
    """
    A 3D triangles mesh.

    Attributes:
        vertices: `array_like`.
            Array of triangle vertices: float (x, y, z) coordinate triplets.
        indices: `array_like`.
            Array of vertex indices: int triplets of indices from vertices array.
        normals: `array_like`.
            Array of vertex normals: float (x, y, z) coordinate triples. Normals are used when flat_shading is false.
            If the normals are not specified here, normals will be automatically computed.
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
        shininess: `float`.
            Shininess of object material.
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
    normals = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("normals")
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
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
    side = TimeSeries(Unicode()).tag(sync=True)
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    volume = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("volume"))
    volume_bounds = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("volume_bounds")
    )
    texture = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("texture")
    )
    texture_file_format = Unicode(allow_none=True).tag(sync=True)
    uvs = TimeSeries(Array()).tag(sync=True, **array_serialization_wrap("uvs"))
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    slice_planes = TimeSeries(ListOrArray(empty_ok=True)).tag(sync=True)
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


# noinspection PyShadowingNames
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
        shininess: `float`.
            Shininess of object material.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = AsciiStlData(allow_none=True, default_value=None).tag(sync=True)
    binary = BinaryStlData(allow_none=True, default_value=None).tag(
        sync=True, **array_serialization_wrap("binary")
    )
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
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
        shininess: `float`.
            Shininess of object material.
        attribute: `array_like`.
            Array of float attribute for the color mapping, coresponding to each vertex.
        opacity: `float`.
            Opacity of surface.
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
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
    attribute = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("attribute")
    )
    opacity = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
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
        from ..helpers import get_bounding_box

        return get_bounding_box(self.model_matrix)
