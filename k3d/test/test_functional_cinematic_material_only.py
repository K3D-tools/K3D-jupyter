"""An edit that only touches a material must not rebuild the traced scene.

A headless sync addresses its diff with the object's id and type, and materialOnly demanded that
every key be a material key, so a color_range change from Python invalidated the proxy and rebuilt
the scene - and the BVH - on every frame of an animation. The tracer keeps the Scene it was last
given, and buildScene makes a fresh one per build, so the scene's uuid says whether a rebuild
happened.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import REF_SAMPLES, prepare

VOLUME = np.random.default_rng(3).random((8, 8, 8), dtype=np.float32)


def _scene_uuid():
    return pytest.headless.browser.execute_script(
        "var t = window.__k3dTracer; return t && t.scene ? t.scene.uuid : null;"
    )


def _traced_frame():
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.get_screenshot(True)


def test_material_edit_keeps_the_traced_scene():
    prepare()
    pytest.plot.renderer = "cinematic"
    pytest.plot.cinematic_samples = REF_SAMPLES
    volume = k3d.volume(VOLUME, color_range=[0.2, 0.8])
    pytest.plot += volume
    _traced_frame()

    before = _scene_uuid()
    assert before is not None

    volume.color_range = [0.1, 0.9]
    _traced_frame()

    assert _scene_uuid() == before

    # the control: a new object is a scene change, and has to rebuild
    pytest.plot += k3d.points(np.array([[2, 2, 2]], dtype=np.float32), point_size=0.2)
    _traced_frame()

    assert _scene_uuid() != before
