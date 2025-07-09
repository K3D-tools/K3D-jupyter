"""Vector objects for K3D."""

import numpy as np
from traitlets import Bool, Float, Int, List, TraitError, Unicode, validate
from traittypes import Array

from .base import Drawable, TimeSeries
from ..helpers import (array_serialization_wrap, get_bounding_box,
                       get_bounding_box_points)


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
    head_size = Float(min=1e-6, default_value=1.0).tag(sync=True)
    scale = Float().tag(sync=True)
    line_width = Float(min=1e-6, default_value=0.01).tag(sync=True)
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
    head_size = Float(min=1e-6, default_value=1.0).tag(sync=True)
    labels = List(default_value=[]).tag(sync=True)
    label_size = Float(min=1e-6, default_value=1.0).tag(sync=True)
    line_width = Float(min=1e-6, default_value=0.01).tag(sync=True)
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
