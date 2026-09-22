"""custom_data is the user's metadata and must survive the trip to the browser unread.

The deserializer used to flag every dict with numeric keys as a time series. An empty dict has
no keyframes, so the interpolator indexed keypoints[0] of nothing and the loader gave up on the
object; a dict keyed by integers grew a time axis the scene never had.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

POSITIONS = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _loaded(obj):
    return pytest.headless.browser.execute_script(
        "return Boolean(K3DInstance.getObjectById(arguments[0]));", obj.id
    )


def test_empty_custom_data_loads():
    prepare()
    obj = k3d.points(POSITIONS, custom_data={})
    pytest.plot += obj
    pytest.headless.sync(hold_until_refreshed=True)

    assert _loaded(obj)


def test_numeric_keys_in_custom_data_are_not_a_time_axis():
    prepare()
    obj = k3d.points(POSITIONS, custom_data={0: "background", 1: "bone"})
    pytest.plot += obj
    pytest.headless.sync(hold_until_refreshed=True)

    assert _loaded(obj)

    # nothing on the scene is animated, so the browser has no time points to offer
    times = pytest.headless.browser.execute_script("""
        var world = K3DInstance.getWorld();
        var times = new Set();

        Object.keys(world.ObjectsListJson).forEach(function (id) {
            var json = world.ObjectsListJson[id];

            Object.keys(json).forEach(function (property) {
                if (property !== 'custom_data' && json[property]
                    && typeof json[property].timeSeries !== 'undefined') {
                    Object.keys(json[property]).forEach(function (t) { times.add(t); });
                }
            });
        });

        return Array.from(times);
    """)

    assert times == []
