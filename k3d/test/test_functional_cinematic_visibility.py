"""Hiding an object has to reach the renderers that keep a scene of their own.

The raster path draws K3DObjects directly, so removing a node from it is the whole story. The
cinematic renderer does not: it mirrors K3DObjects into a proxy scene, builds a BVH over that, and
only rebuilds when an object event says something changed. Core.reload() has an early return for
visible === false that removes the node and used to leave without announcing it, so the traced
image kept the object that was no longer in the scene.

Counting triangles in the tracer's own last build says this directly, and for any object type,
where a pixel comparison would only say that two images differ.
"""
import numpy as np
import pytest

import k3d

from .plot_compare import prepare

BUILD = "return window.K3DInstance.__cinematicSpike().lastBuild();"

# a quad and a single triangle: distinct counts, so the assertion can name what left
QUAD = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
QUAD_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
TRI = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]], dtype=np.float32)
TRI_FACES = np.array([[0, 1, 2]], dtype=np.uint32)


def _traced_triangles():
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.get_screenshot(True)

    return pytest.headless.browser.execute_script(BUILD)["triangles"]


def test_hiding_an_object_takes_it_out_of_the_traced_scene():
    prepare()
    plot = pytest.plot

    keeper = k3d.mesh(TRI, TRI_FACES, color=0x3F6BFA)
    hidden = k3d.mesh(QUAD, QUAD_FACES, color=0xE6006E)

    plot += keeper
    plot += hidden
    plot.renderer = "cinematic"
    plot.cinematic_samples = 1

    both = _traced_triangles()
    assert both >= 3, "the traced scene does not hold the two meshes (%d triangles)" % both

    hidden.visible = False
    without = _traced_triangles()
    assert without == both - 2, (
        "hiding the quad left %d triangles in the traced scene against %d before: the renderer was "
        "never told, so it keeps tracing an object that is no longer in the scene"
        % (without, both))

    # and back: this direction always worked, since the loader path announces itself
    hidden.visible = True
    again = _traced_triangles()
    assert again == both, (
        "showing the quad again left %d triangles against %d" % (again, both))
