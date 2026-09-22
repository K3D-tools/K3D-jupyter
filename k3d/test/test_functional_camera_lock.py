"""camera_no_rotate / camera_no_zoom / camera_no_pan have to hold in every camera mode, and
survive the controls being rebuilt.

setCameraLock wrote the TrackballControls flags (noRotate, ...) while OrbitControls reads its own
(enableRotate, ...), so an orbit camera ignored the lock; and Canvas rebuilt the controls without
the locks whenever damping, up axis or camera mode changed, so a locked trackball came unlocked
the moment a damping factor was set.

The drag is a real pointer gesture through Selenium, so what is asserted is what a user sees.
"""

import time

import numpy as np
import pytest
from selenium.webdriver.common.action_chains import ActionChains

import k3d

from .plot_compare import prepare

POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _camera_position():
    return pytest.headless.browser.execute_script(
        "var c = K3DInstance.getWorld().camera.position; return [c.x, c.y, c.z];"
    )


def _drag(dx=120, dy=60):
    canvas = pytest.headless.browser.find_element("css selector", "#canvasTarget canvas")
    chain = ActionChains(pytest.headless.browser)
    chain.move_to_element(canvas).click_and_hold().move_by_offset(dx, dy).release().perform()
    # the controls apply the gesture on their animation frame, not on the event
    time.sleep(0.4)


def _scene(camera_mode):
    prepare()
    pytest.plot += k3d.points(POSITIONS, point_size=0.2)
    pytest.plot.camera_mode = camera_mode
    pytest.headless.sync(hold_until_refreshed=True)


def _set_lock(on):
    pytest.plot.camera_no_rotate = on
    pytest.plot.camera_no_zoom = on
    pytest.plot.camera_no_pan = on
    pytest.headless.sync(hold_until_refreshed=True)


@pytest.mark.parametrize("camera_mode", ["trackball", "orbit"])
def test_lock_holds_the_camera_still(camera_mode):
    _scene(camera_mode)

    # the control: the gesture does move an unlocked camera in this mode
    before = _camera_position()
    _drag()
    assert not np.allclose(_camera_position(), before, atol=1e-6)

    _set_lock(True)
    before = _camera_position()
    _drag()

    assert np.allclose(_camera_position(), before, atol=1e-6)

    _set_lock(False)


def test_lock_survives_rebuilt_controls():
    _scene("trackball")
    _set_lock(True)

    # each of these makes Canvas throw the controls away and build fresh ones
    pytest.plot.camera_damping_factor = 0.2
    pytest.headless.sync(hold_until_refreshed=True)

    before = _camera_position()
    _drag()

    assert np.allclose(_camera_position(), before, atol=1e-6)

    pytest.plot.camera_damping_factor = 0.0
    _set_lock(False)
