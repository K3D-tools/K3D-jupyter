"""A cinematic scene traces all of its triangles, whatever was traced before it.

three-mesh-bvh 0.9.15 reuses the previous scene's index when its length equals the new
triangle count; the backend drops it before every build.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare


def _triangles(count, size):
    """`count` separate triangles in a square grid in the x-z plane, facing the camera."""
    side = int(np.ceil(np.sqrt(count)))
    vertices = []
    for i in range(count):
        x, z = (i % side) * size, (i // side) * size
        vertices += [[x, 0, z], [x + 0.9 * size, 0, z], [x, 0, z + 0.9 * size]]
    return np.array(vertices, np.float32), np.arange(3 * count, dtype=np.uint32).reshape(-1, 3)


def _coverage():
    pytest.plot.renderer = "cinematic"
    pytest.plot.cinematic_samples = 4
    pytest.plot.screenshot_scale = 0.5
    try:
        pytest.headless.sync(hold_until_refreshed=True)
        image = np.asarray(Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGB"))
    finally:
        pytest.plot.screenshot_scale = 1.0
        pytest.plot.renderer = "simple"

    # the background is white, the triangles dark
    return float((image.min(axis=2) < 200).mean())


def _scene(count):
    prepare()
    vertices, indices = _triangles(count, 1.0)
    pytest.plot += k3d.mesh(vertices, indices, color=0x203040, side="double")
    pytest.plot.camera = [2.5, -12, 2.5, 2.5, 0, 2.5, 0, 0, 1]


def test_every_triangle_is_traced_after_a_smaller_scene():
    _scene(36)
    alone = _coverage()

    # 36 indices before 36 triangles: the case the cached index got wrong
    _scene(12)
    _coverage()
    _scene(36)
    after = _coverage()

    assert alone > 0.02
    assert after == pytest.approx(alone, rel=0.02)
