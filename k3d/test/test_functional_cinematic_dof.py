"""Depth of field must cost nothing until it is asked for.

The cinematic renderer's camera is a PhysicalCamera so that the path tracer will read a lens off
it - its uniform copies nothing from a camera that is not an instance of that class. But that
class defaults to f/1.4, which is an aperture of a dozen scene units or more, so the whole of the
visual reference suite rests on one thing: FEATURE_DOF stays 0 while cinematic_bokeh_size is 0,
and the shader compiles without the aperture code at all.

Read from the material rather than from pixels: the define is the invariant, and a pixel
comparison would only say that two images look the same at whatever sample count the test can
afford.
"""
import numpy as np
import pytest

import k3d

from .plot_compare import prepare

STATE = """
var m = window.__k3dTracer._pathTracer.material;
var c = window.K3DInstance.getWorld().camera;

return {
    dof: m.defines.FEATURE_DOF,
    uniformBokeh: m.physicalCamera.bokehSize,
    uniformFocus: m.physicalCamera.focusDistance,
    uniformBlades: m.physicalCamera.apertureBlades,
    cameraIsPhysical: typeof c.fStop !== 'undefined'
};
"""

VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def _render_and_read():
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.get_screenshot(True)

    return pytest.headless.browser.execute_script(STATE)


def test_depth_of_field_is_off_until_the_aperture_opens():
    prepare()
    plot = pytest.plot
    plot += k3d.mesh(VERTICES, INDICES, color=0x3F6BFA)
    plot.renderer = "cinematic"
    plot.cinematic_samples = 1

    closed = _render_and_read()
    assert closed["cameraIsPhysical"], (
        "the world camera is not a PhysicalCamera, so the tracer cannot read a lens off it at all")
    assert closed["dof"] == 0, (
        "FEATURE_DOF is %s with the aperture closed: the depth of field code is compiled into "
        "every cinematic render, and every reference image in the suite is at its mercy"
        % closed["dof"])
    assert closed["uniformBokeh"] == 0

    # the camera's own target, which is what a focus distance of 0 means
    plot.cinematic_bokeh_size = 0.5
    plot.cinematic_aperture_blades = 6
    opened = _render_and_read()

    assert opened["dof"] == 1, "the aperture opened and the shader did not follow"
    # the parameter is a diameter in scene units; the shader wants millimetres against metres
    assert opened["uniformBokeh"] == pytest.approx(500.0)
    assert opened["uniformBlades"] == 6
    assert opened["uniformFocus"] > 0.0, (
        "focus distance resolved to %s - with the parameter left at 0 it should follow the "
        "camera's target" % opened["uniformFocus"])

    # an iris of one or two sides is not a shape, and must read as a circle rather than as noise
    plot.cinematic_aperture_blades = 0
    assert _render_and_read()["uniformBlades"] == 0

    plot.cinematic_bokeh_size = 0.0
    back = _render_and_read()
    assert back["dof"] == 0, "closing the aperture did not put the shader back"
    assert back["uniformBokeh"] == 0


def test_the_compile_a_volume_plus_aperture_needs_is_not_read_as_a_stall():
    """The watchdog counts turns, and a compile spends them without advancing a sample.

    PathTracingRenderer.update() returns on _compilePromise and WebGLPathTracer.renderSample()
    gates on the same flag, so while a program links the sample counter sits at exactly 0 -
    which is what a stall looks like from outside. The ceiling was 5000 turns of a setTimeout
    the browser clamps to 4 ms, i.e. ~20 s, against ~16 s for the volume shader alone on a
    laptop 4070; opening the aperture makes FEATURE_DOF 1, which is a different program again,
    and the first render of a volume with a lens died on its own guard.

    This renders the combination that failed. It does not pin the timing - the compile here is
    whatever this machine's driver costs, and on a fast one the old code would have survived
    too - so read it as a guard on the combination, not as a reproduction of the ceiling.
    """
    prepare()
    plot = pytest.plot

    n = 24
    grid = (np.mgrid[0:n, 0:n, 0:n] + 0.5) / n - 0.5
    ball = np.where(np.sqrt((grid ** 2).sum(axis=0)) < 0.4, 1.0, 0.0).astype(np.float32)

    plot += k3d.volume(ball, color_range=[0.0, 1.0], alpha_coef=30.0,
                       bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    plot.renderer = "cinematic"
    plot.cinematic_samples = 1
    plot.cinematic_bokeh_size = 0.5

    state = _render_and_read()

    assert state["dof"] == 1, "the aperture is open and the volume shader did not get the lens"
    assert state["uniformBokeh"] == pytest.approx(500.0)
