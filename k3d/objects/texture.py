"""Texture objects for K3D."""

import numpy as np
from traitlets import (
    Bool,
    Bytes,
    Float,
    Unicode,
)
from traittypes import Array

from ..helpers import (
    array_serialization_wrap,
    get_bounding_box,
)
from .base import (
    DrawableWithCallback,
    TimeSeries,
    ListOrArray,
)


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
    binary = Bytes(allow_none=True).tag(
        sync=True, **array_serialization_wrap("binary"))
    file_format = Unicode(allow_none=True).tag(sync=True)
    attribute = Array().tag(sync=True, **array_serialization_wrap("attribute"))
    puv = Array(dtype=np.float32).tag(
        sync=True, **array_serialization_wrap("puv"))
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