"""Points objects for K3D."""

import numpy as np
from traitlets import Float, Int, TraitError, Unicode, validate
from traittypes import Array

from .base import EPSILON, Drawable, ListOrArray, TimeSeries
from ..helpers import array_serialization_wrap, get_bounding_box_points


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
        shininess: `float`.
            Shininess of object material.
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
    shininess = TimeSeries(Float(default_value=50.0)).tag(sync=True)
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
