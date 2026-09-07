"""A guide for the denoiser must not become part of the image it guides.

The variance buffer splits the accumulation by sample parity and hands the two halves to the
filter, which reads their difference per pixel. It never enters the tracing program - it blends a
texture the tracer already wrote - so the traced image has no way to notice, and that is the
property worth pinning: a guide that moves a single pixel would put all 150 cinematic references
at the mercy of a debug flag.

The proof has to run on the interactive path. renderBudget - the screenshot path - resizes the
tracer, and a test built on get_screenshot would compare two renders under a different setup than
the one the viewport uses. Driving renderSamplesAsync through the diagnostic handle presents to
the canvas instead, and preserveDrawingBuffer is on (Renderer.js), so the canvas can be read back.

The second half checks the contract the filter depends on: nothing until both halves hold a
sample, because one alone says nothing about spread, and then the weight that turns their
difference into the variance of the mean.
"""
import time

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

START = """
window.__varianceProbe = null;

var mode = K3DInstance.__cinematicSpike();

mode.setVariance(arguments[0], true);
mode.renderSamplesAsync(arguments[1]).then(function (result) {
    window.__varianceProbe = {
        canvas: K3DInstance.getWorld().renderer.domElement.toDataURL(),
        samples: result.samples
    };
}, function (e) {
    window.__varianceProbe = { error: String(e) };
});
"""

POLL = "return window.__varianceProbe;"
# scalars only: the halves themselves are THREE textures and do not survive the
# WebDriver JSON protocol
HALVES = """
var h = K3DInstance.__cinematicSpike().varianceHalves();

return h === null ? null : { weight: h.weight, samples: h.samples, ready: !!(h.a && h.b) };
"""

VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                     [0, 0, 1.2]], dtype=np.float32)
INDICES = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4]], dtype=np.uint32)


def _wait(script, seconds=120.0):
    deadline = time.time() + seconds

    while time.time() < deadline:
        value = pytest.headless.browser.execute_script(script)

        if value is not None:
            return value

        time.sleep(0.1)

    raise AssertionError("the browser did not answer %s within %.0f s" % (script, seconds))


def _render(samples, variance):
    pytest.headless.browser.execute_script(START, variance, samples)

    result = _wait(POLL)

    assert "error" not in result, "the accumulation failed: %s" % result.get("error")
    assert result["samples"] == samples, (
        "asked for %d samples and the tracer stopped at %s" % (samples, result["samples"]))

    return result["canvas"]


def test_the_guide_does_not_change_the_image():
    prepare()
    plot = pytest.plot
    plot += k3d.mesh(VERTICES, INDICES, color=0x3F6BFA, roughness=0.4)
    plot.renderer = "cinematic"
    plot.cinematic_seed = 1
    pytest.headless.sync(hold_until_refreshed=True)

    try:
        off = _render(16, False)
        on = _render(16, True)

        assert on == off, (
            "the canvas differs with the variance buffer on: a guide is writing into the traced "
            "image, and every cinematic reference now depends on a debug flag")

        halves = pytest.headless.browser.execute_script(HALVES)

        assert halves is not None, "16 samples in and the halves are still not offered"
        assert halves["ready"], "the halves are offered without textures behind them"
        assert halves["samples"] == 16, (
            "the halves report %s samples against the 16 that were traced" % halves["samples"])
        # both halves hold eight, so nA * nB / n^2 is exactly a quarter
        assert halves["weight"] == pytest.approx(0.25), (
            "the weight is %s: it converts the half-difference into the variance of the mean and "
            "is nA * nB / n^2, which at an even split is 1/4" % halves["weight"])

        # one sample fills one half, and a single half says nothing about spread
        _render(1, True)

        assert pytest.headless.browser.execute_script(HALVES) is None, (
            "the halves were offered after a single sample, when only one of them holds anything "
            "- a filter guided by that difference would read the image itself as noise")
    finally:
        pytest.headless.browser.execute_script(
            "K3DInstance.__cinematicSpike().setVariance(false);")
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)
