"""Text objects for K3D."""

import numpy as np
from traitlets import (
    Bool,
    Float,
    Int,
    Unicode,
)
from traittypes import Array

from .base import (
    Drawable,
    TimeSeries,
    SingleOrList,
    EPSILON,
)
from ..helpers import (
    array_serialization_wrap,
    get_bounding_box_point,
)


class Text(Drawable):
    """
    Text rendered using KaTeX with a 3D position.

    Attributes:
        text: str or list of str
            Content of the text.
        position : list
            (x, y, z) coordinates of text position, by default (0, 0, 0).
            If n text is pass position should contain 3*n elements .
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
    text = TimeSeries(SingleOrList(Unicode())).tag(sync=True)
    position = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                       **array_serialization_wrap("position"))
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
        text: str or list of str
            Content of the text.
        position: `list`.
            Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size.
            If n text is pass position should contain 2*n elements .
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
    text = TimeSeries(SingleOrList(Unicode())).tag(sync=True)
    position = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                       **array_serialization_wrap("position"))
    is_html = Bool(False).tag(sync=True)
    color = Int(min=0, max=0xFFFFFF).tag(sync=True)
    reference_point = Unicode().tag(sync=True)
    size = TimeSeries(Float(min=EPSILON, default_value=1.0)).tag(sync=True)
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
        text: str or list of str
            Content of the text.
        position : list
            (x, y, z) coordinates of text position, by default (0, 0, 0).
            If n text is pass position should contain 3*n elements .
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
    text = TimeSeries(SingleOrList(Unicode())).tag(sync=True)
    is_html = Bool(False).tag(sync=True)
    position = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                       **array_serialization_wrap("position"))
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


class TextureText(Drawable):
    """
    A text in the 3D space rendered using a texture.

    Compared to Text and Text2d this drawable has less features (no KaTeX support), but the labels are located
    in the GPU memory, and not the browser's DOM tree. This has performance consequences, and may be preferable when
    many simple labels need to be displayed.

    Attributes:
        text: str or list of str
            Content of the text.
        position : list
            (x, y, z) coordinates of text position, by default (0, 0, 0).
            If n text is pass position should contain 3*n elements .
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
    text = TimeSeries(SingleOrList(Unicode())).tag(sync=True)
    position = TimeSeries(Array(dtype=np.float32)).tag(sync=True,
                                                       **array_serialization_wrap("position"))
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
