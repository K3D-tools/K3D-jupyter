"""A sync applies the plot's parameters to the scene the objects in that same sync produce.

headless.html used to apply plot_diff before objects_diff, so the first sync of a session set the
camera on an empty scene - and with camera_auto_fit on, the objects arriving afterwards refitted
it, discarding the camera the user had just asked for. A screenshot straight after that sync had
the wrong view; every probe in the repository learnt to sync twice.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

POSITIONS = np.array([[-3, -3, -3], [3, 3, 3]], dtype=np.float32)
CAMERA = [10.0, -20.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _camera_position():
    return pytest.headless.browser.execute_script(
        "var c = K3DInstance.getWorld().camera.position; return [c.x, c.y, c.z];"
    )


def test_camera_set_alongside_the_object_it_frames_is_kept():
    prepare()

    # one sync carries all three: auto-fit on, the object, the explicit camera
    pytest.plot.camera_auto_fit = True
    pytest.plot += k3d.points(POSITIONS, point_size=0.3)
    pytest.plot.camera = CAMERA
    pytest.headless.sync(hold_until_refreshed=True)

    assert np.allclose(_camera_position(), CAMERA[:3], atol=1e-3)

    pytest.plot.camera_auto_fit = False


def test_time_set_alongside_the_object_it_animates_is_kept():
    """Same ordering, other parameter: a time arriving with the object it animates."""
    prepare()

    frame_a = np.array([[0, 0, 0]], dtype=np.float32)
    frame_b = np.array([[0, 0, 5]], dtype=np.float32)
    obj = k3d.points({"0.0": frame_a, "1.0": frame_b}, point_size=0.3)

    pytest.plot += obj
    pytest.plot.time = 1.0
    pytest.headless.sync(hold_until_refreshed=True)

    position = pytest.headless.browser.execute_script(
        "var o = K3DInstance.getObjectById(arguments[0]);"
        "var p = o.geometry.attributes.position.array; return [p[0], p[1], p[2]];",
        obj.id,
    )

    assert np.allclose(position, frame_b[0], atol=1e-6)

    pytest.plot.time = 0.0
