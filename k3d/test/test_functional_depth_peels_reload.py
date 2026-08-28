"""Changing depth_peels has to rebuild every object, animated ones included.

Blending and the peel shader hook are chosen when a material is built, so an object left
configured for the other pipeline renders opaque whatever its opacity says. An update cannot
reconfigure it, and a time-series object always has an interpolated change to resolve - which
is exactly how it used to keep the old material.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

MATERIAL = """
var out = null;

K3DInstance.getWorld().K3DObjects.traverse(function (node) {
    if (node.material && out === null) {
        out = {blending: node.material.blending, uuid: node.material.uuid};
    }
});

return out;
"""


def _material():
    return pytest.headless.browser.execute_script(MATERIAL)


def _points(animated):
    positions = np.random.RandomState(3).randn(64, 3).astype(np.float32)
    points = k3d.points(positions, color=0x0000CC, shader='mesh', point_size=0.4)

    pytest.plot += points

    if animated:
        points.positions = {
            str(i / 10): positions + 0.04 * np.random.RandomState(i).randn(64, 3).astype(np.float32)
            for i in range(4)
        }

    pytest.headless.sync(hold_until_refreshed=True)

    return points


def test_depth_peels_rebuilds_an_animated_object():
    prepare()

    _points(animated=True)
    before = _material()

    pytest.plot.depth_peels = 5
    pytest.headless.sync(hold_until_refreshed=True)
    after = _material()

    assert after['uuid'] != before['uuid'], 'the material was not rebuilt'
    assert after['blending'] != before['blending'], (before, after)


def test_depth_peels_rebuilds_a_static_object():
    prepare()

    _points(animated=False)
    before = _material()

    pytest.plot.depth_peels = 5
    pytest.headless.sync(hold_until_refreshed=True)
    after = _material()

    assert after['uuid'] != before['uuid'], 'the material was not rebuilt'
    assert after['blending'] != before['blending'], (before, after)
