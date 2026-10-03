"""Geometric objects for K3D."""

import warnings

import numpy as np
from traitlets import Bool, Bytes, TraitError, Unicode, validate

from ..helpers import Array, Float, Int, array_serialization_wrap, get_bounding_box_points
from ..validation.stl import (
    AsciiStlData,
    BinaryStlData,
    vertices_from_ascii,
    vertices_from_binary,
)
from .base import (
    EPSILON,
    Drawable,
    DrawableWithCallback,
    ListOrArray,
    TimeSeries,
    resolve_color,
)

TEXTURE_WRAPS = ("clamp", "repeat", "mirror")
ALPHA_MODES = ("opaque", "blend", "mask")


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
        roughness: `float`.
            Roughness of object material.
        metalness: `float`.
            Metalness of object material.
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
    roughness = TimeSeries(Float(default_value=0.4, min=0.0, max=1.0)).tag(sync=True)
    metalness = TimeSeries(Float(default_value=0.0, min=0.0, max=1.0)).tag(sync=True)
    radial_segments = TimeSeries(Int()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                "colors has wrong size: %s (%s required, one per vertex rather than per segment)"
                % (actual, required)
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
        roughness: `float`.
            Roughness of object material.
        metalness: `float`.
            Metalness of object material.
        radial_segments: 'int':
            Number of segmented faces around the circumference of the tube.
        model_matrix: `array_like`.
            4x4 model transform matrix.
    """

    type = Unicode(read_only=True).tag(sync=True)

    vertices = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("vertices")
    )
    indices = Array(dtype=np.uint32).tag(
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
    roughness = TimeSeries(Float(default_value=0.4, min=0.0, max=1.0)).tag(sync=True)
    metalness = TimeSeries(Float(default_value=0.0, min=0.0, max=1.0)).tag(sync=True)
    radial_segments = TimeSeries(Int()).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                "colors has wrong size: %s (%s required, one per vertex rather than per segment)"
                % (actual, required)
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
            Packed RGB color of the mesh (0xff0000 is red, 0xff is blue). It multiplies `colors`,
            the colormap and `texture`; left out, it is white when any of them is given.
        colors: `array_like`.
            Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is blue).
        opacities: `array_like`.
            Same-length array of `float` alpha per vertex, multiplied into the colour. Read when
            `alpha_mode` is 'blend' or 'mask'.
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
        roughness: `float`.
            Roughness of object material.
        metalness: `float`.
            Metalness of object material.
        opacity: `float`.
            Opacity of mesh.
        volume: `array_like`.
            3D array of `float`, indexed as [z, y, x].
        volume_bounds: `array_like`.
            6-element tuple specifying the bounds of the volume data (x0, x1, y0, y1, z0, z1)
        texture: `bytes`.
            Image data in a specific format.
        texture_file_format: `str`.
            Format of the data, it should be the second part of MIME format of type 'image/',
            for example 'jpeg', 'png', 'gif', 'tiff'.
        uvs: `array_like`.
            Array of float uvs for the texturing, coresponding to each vertex.
        uvs2: `array_like`.
            A second set of uvs, read by `occlusion_map` alone; it falls back to `uvs`.
        texture_wrap: `str`.
            What the textures do outside 0..1: 'clamp' (the default), 'repeat' or 'mirror' - or
            one for u and one for v, e.g. 'repeat clamp'.
        emissive: `int`.
            Packed RGB color the surface emits, unaffected by lighting (0 is none).
        emissive_intensity: `float`.
            Multiplier of `emissive`.
        emissive_map: `bytes`.
            Image multiplying `emissive`, in PNG, JPEG, WebP or GIF.
        normal_map: `bytes`.
            Tangent-space normal map image, read with `uvs`.
        normal_scale: `float`.
            Strength of `normal_map`.
        metalness_roughness_map: `bytes`.
            Image whose green channel multiplies `roughness` and blue channel `metalness`.
        occlusion_map: `bytes`.
            Image whose red channel darkens indirect light, read with `uvs2` when given.
        occlusion_strength: `float`.
            How much of `occlusion_map` applies, 0 to 1.
        alpha_mode: `str`.
            How the alpha of `texture` and `opacities` is used:

            :`opaque`: ignored, only `opacity` fades the mesh (the default),

            :`blend`: blended with what is behind,

            :`mask`: cut out where it is below `alpha_cutoff`, the rest is solid.
        alpha_cutoff: `float`.
            Threshold of the 'mask' mode.
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
    roughness = TimeSeries(Float(default_value=0.4, min=0.0, max=1.0)).tag(sync=True)
    metalness = TimeSeries(Float(default_value=0.0, min=0.0, max=1.0)).tag(sync=True)
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
    uvs = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("uvs")
    )
    opacity_function = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacity_function")
    )
    slice_planes = TimeSeries(ListOrArray(empty_ok=True)).tag(sync=True)
    opacities = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("opacities")
    )
    uvs2 = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("uvs2")
    )
    texture_wrap = Unicode("clamp").tag(sync=True)
    emissive = TimeSeries(Int(min=0, max=0xFFFFFF)).tag(sync=True)
    emissive_intensity = TimeSeries(Float(min=0.0, default_value=1.0)).tag(sync=True)
    emissive_map = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("emissive_map")
    )
    normal_map = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("normal_map")
    )
    normal_scale = TimeSeries(Float(default_value=1.0)).tag(sync=True)
    metalness_roughness_map = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("metalness_roughness_map")
    )
    occlusion_map = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("occlusion_map")
    )
    occlusion_strength = TimeSeries(Float(min=0.0, max=1.0, default_value=1.0)).tag(sync=True)
    alpha_mode = Unicode("opaque").tag(sync=True)
    alpha_cutoff = TimeSeries(Float(min=0.0, max=1.0, default_value=0.5)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        resolve_color(kwargs, ("colors", "attribute", "triangles_attribute", "texture"))

        super().__init__(**kwargs)

        self.set_trait("type", "Mesh")

    @validate("texture_wrap")
    def _validate_texture_wrap(self, proposal):
        modes = proposal["value"].split()
        if not 1 <= len(modes) <= 2 or any(mode not in TEXTURE_WRAPS for mode in modes):
            raise TraitError("texture_wrap must be one of %s, or two of them for u and v "
                             "('repeat clamp'), not %r" % (", ".join(TEXTURE_WRAPS), proposal["value"]))
        return proposal["value"]

    @validate("alpha_mode")
    def _validate_alpha_mode(self, proposal):
        if proposal["value"] not in ALPHA_MODES:
            raise TraitError("alpha_mode must be one of %s, not %r"
                             % (", ".join(ALPHA_MODES), proposal["value"]))
        return proposal["value"]

    @validate("opacities")
    def _validate_opacities(self, proposal):
        if type(proposal["value"]) is dict or type(self.vertices) is dict:
            return proposal["value"]

        required = self.vertices.size // 3
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "opacities has wrong size: %s (%s required, one per vertex)" % (actual, required)
            )
        return proposal["value"]

    @validate("colors")
    def _validate_colors(self, proposal):
        if type(proposal["value"]) is dict or type(self.vertices) is dict:
            return proposal["value"]

        required = self.vertices.size // 3  # (x, y, z) triplet per 1 color
        actual = proposal["value"].size
        if actual != 0 and required != actual:
            raise TraitError(
                "colors has wrong size: %s (%s required, one per vertex)" % (actual, required)
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
            warnings.warn("wrong dtype: %s (%s required)" % (actual, required),
                          stacklevel=2)

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
        roughness: `float`.
            Roughness of object material.
        metalness: `float`.
            Metalness of object material.
    """

    type = Unicode(read_only=True).tag(sync=True)
    text = AsciiStlData(allow_none=True, default_value=None).tag(sync=True)
    binary = BinaryStlData(allow_none=True, default_value=None).tag(
        sync=True, **array_serialization_wrap("binary")
    )
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    wireframe = Bool().tag(sync=True)
    flat_shading = Bool().tag(sync=True)
    roughness = TimeSeries(Float(default_value=0.4, min=0.0, max=1.0)).tag(sync=True)
    metalness = TimeSeries(Float(default_value=0.0, min=0.0, max=1.0)).tag(sync=True)
    model_matrix = TimeSeries(Array(dtype=np.float32)).tag(
        sync=True, **array_serialization_wrap("model_matrix")
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_trait("type", "STL")

    def get_bounding_box(self):
        if self.text is not None:
            vertices = vertices_from_ascii(self.text)
        elif self.binary is not None:
            vertices = vertices_from_binary(self.binary)
        else:
            vertices = np.zeros((0, 3), dtype=np.float32)

        return get_bounding_box_points(vertices, self.model_matrix)


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
        roughness: `float`.
            Roughness of object material.
        metalness: `float`.
            Metalness of object material.
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
    roughness = TimeSeries(Float(default_value=0.4, min=0.0, max=1.0)).tag(sync=True)
    metalness = TimeSeries(Float(default_value=0.0, min=0.0, max=1.0)).tag(sync=True)
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
        super().__init__(**kwargs)

        self.set_trait("type", "Surface")

    def get_bounding_box(self):
        from ..helpers import _flatten_frames, get_bounding_box

        # the renderer puts a vertex at the raw height; only x and y come from the matrix
        heights = _flatten_frames(self.heights)
        boundary = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]

        if heights.shape[0] > 0:
            boundary[4] = float(np.nanmin(heights))
            boundary[5] = float(np.nanmax(heights))

        return get_bounding_box(self.model_matrix, boundary)
