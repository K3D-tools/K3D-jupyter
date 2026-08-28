"""A silent camera update must not ask for a render - in orbit mode as well.

The time series animates the camera through setupCamera(..., silent=true) and renders once per
frame itself; the flag exists so the controls do not announce the same move a second time. In
OrbitControls the flag was dead: it was declared on a factory that runs once with no arguments,
while the exposed function is the inner one, so every silent update still dispatched 'change'
and drove a redundant render request.
"""
import numpy as np
import pytest

import k3d

from .plot_compare import prepare

MEASURE = """
var world = K3DInstance.getWorld();

// switching the mode rebuilds the controls, refits the scene and - in orbit mode - rewrites the
// camera up vector, and resetCamera() in the next test keeps the direction it finds. Put the whole
// camera back or every later visual test renders from here.
var before = world.camera.position.toArray()
    .concat(world.controls.target.toArray())
    .concat(world.camera.up.toArray());

K3DInstance.setCameraMode(arguments[0]);

var inner = world.render.bind(world);
var n = 0;

world.render = function (force) { n += 1; return inner(force); };

// the camera has to move, otherwise the controls have nothing to announce either way
world.setupCamera([6, 0, 2, 0, 0, 0, 0, 0, 1], null, true);
var silent = n;

world.setupCamera([0, 6, 2, 0, 0, 0, 0, 0, 1], null, false);
var loud = n - silent;

world.render = inner;
K3DInstance.setCameraMode('trackball');
world.setupCamera(before, null, true);

return [silent, loud];
"""


@pytest.mark.parametrize('mode', ['trackball', 'orbit'])
def test_a_silent_camera_update_asks_for_no_render(mode):
    prepare()

    pytest.plot += k3d.points(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), point_size=0.2)
    pytest.headless.sync(hold_until_refreshed=True)

    silent, loud = pytest.headless.browser.execute_script(MEASURE, mode)

    assert silent == 0, 'a silent update asked for %d renders in %s mode' % (silent, mode)
    assert loud >= 1, 'a loud update has to ask for one, %s mode' % mode


def test_camera_mode_reaches_the_page_from_python():
    """camera_mode used to be missing from the headless trait map.

    It is in the anywidget map, so the widget honoured it and the headless harness silently did
    not - which is the trap of a trait needing four registration points. A test that switches the
    mode from Python is the only thing that keeps the fourth one alive.
    """
    prepare()
    pytest.headless.sync(hold_until_refreshed=True)

    before = pytest.headless.browser.execute_script(
        'var w = K3DInstance.getWorld();'
        'return w.camera.position.toArray().concat(w.controls.target.toArray())'
        '    .concat(w.camera.up.toArray());')

    try:
        pytest.plot.camera_mode = 'orbit'
        pytest.headless.sync(hold_until_refreshed=True)

        assert pytest.headless.browser.execute_script(
            'return K3DInstance.parameters.cameraMode') == 'orbit'
    finally:
        pytest.plot.camera_mode = 'trackball'
        pytest.headless.sync(hold_until_refreshed=True)
        # switching the mode refits the scene, and resetCamera() keeps whatever direction it
        # finds, so the camera has to go back or every later visual test renders from here
        pytest.headless.browser.execute_script(
            'K3DInstance.getWorld().setupCamera(arguments[0], null, true);', before)

    assert pytest.headless.browser.execute_script(
        'return K3DInstance.parameters.cameraMode') == 'trackball'
