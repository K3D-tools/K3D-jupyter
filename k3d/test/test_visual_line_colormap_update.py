"""Visual coverage for the colormap update fast paths of line objects.

Two things make these paths easy to get wrong: neither material exposes low/high uniforms, so
a new `color_range` means recomputing the uv attribute, and for `lines()` that attribute is in
deduplicated edge order rather than 1:1 with the input.

Each test renders the object, then changes only `color_range` (or `attribute`) and renders
again, so the second comparison fails if the update path throws, no-ops or mis-maps.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import compare, prepare

VERTICES = np.array(
    [[-1, -1, 0], [-0.5, 1, 0.3], [0, -1, 0.6], [0.5, 1, 0.9], [1, -1, 1.2]],
    dtype=np.float32,
)
ATTRIBUTE = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

# A closed polyline: every vertex appears in two segments, so the edge expansion is NOT the
# identity - which is what makes the uv mapping observable.
SEGMENT_INDICES = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.uint32
)


def _line(shader):
    return k3d.line(
        VERTICES,
        attribute=ATTRIBUTE,
        color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        color_range=[0.0, 1.0],
        shader=shader,
        width=0.08,
    )


def _lines(shader):
    return k3d.lines(
        VERTICES,
        SEGMENT_INDICES,
        indices_type="segment",
        attribute=ATTRIBUTE,
        color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        color_range=[0.0, 1.0],
        shader=shader,
        width=0.08,
    )


def test_line_thick_colormap_color_range_update():
    prepare()
    obj = _line("thick")
    pytest.plot += obj

    compare("line_thick_colormap")

    # Half the range: the colormap must be recomputed, not left as it was.
    obj.color_range = [0.0, 0.5]

    compare("line_thick_colormap_half_range")


def test_line_simple_colormap_color_range_update():
    prepare()
    obj = _line("simple")
    pytest.plot += obj

    compare("line_simple_colormap")

    obj.color_range = [0.0, 0.5]

    compare("line_simple_colormap_half_range")


def test_lines_thick_colormap_color_range_update():
    prepare()
    obj = _lines("thick")
    pytest.plot += obj

    compare("lines_thick_colormap")

    obj.color_range = [0.0, 0.5]

    compare("lines_thick_colormap_half_range")


def test_lines_simple_colormap_color_range_update():
    prepare()
    obj = _lines("simple")
    pytest.plot += obj

    compare("lines_simple_colormap")

    obj.color_range = [0.0, 0.5]

    compare("lines_simple_colormap_half_range")


def test_lines_thick_colormap_attribute_update():
    """Attribute change on an edge-expanded object: exercises the uv mapping directly."""
    prepare()
    obj = _lines("thick")
    pytest.plot += obj

    compare("lines_thick_colormap")

    # Reversed attribute - with a correct mapping the gradient flips along the polyline.
    obj.attribute = ATTRIBUTE[::-1].copy()

    compare("lines_thick_colormap_reversed")


def test_lines_simple_colormap_attribute_update():
    prepare()
    obj = _lines("simple")
    pytest.plot += obj

    compare("lines_simple_colormap")

    obj.attribute = ATTRIBUTE[::-1].copy()

    compare("lines_simple_colormap_reversed")
