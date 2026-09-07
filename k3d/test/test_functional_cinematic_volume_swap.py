"""The majorant grid has to describe the volume the tracer is actually tracking.

Delta tracking is unbiased only while the per-cell bound is at or above the extinction at every
point a sample can land. The grid is cached, so the question is whether the cache can hand one
volume the bounds built from another: only the first volume is traced, and removing it promotes
the next, which is where a cache keyed on values rather than on identity collides - two volumes of
the same shape, colour range and alpha_coef agree on every version number.

The failure is silent in the image (the render is merely too transparent), so this reads the grid
itself: the thin shell and the solid ball cannot share a majorant.
"""
import numpy as np
import pytest

import k3d

from .plot_compare import prepare

GRID = """
var u = window.__k3dTracer._pathTracer.material.uniforms;
var grid = u.volumeMajorant.value;
var d = grid && grid.image ? grid.image.data : null;
var empty = 0;
var sum = 0;

if (d) {
    for (var i = 0; i < d.length; i += 1) {
        if (d[i] <= 0) { empty += 1; } else { sum += d[i]; }
    }
}

return {
    traced: u.volumeTexture.value ? u.volumeTexture.value.id : null,
    cells: d ? d.length : 0,
    empty: empty,
    mean: d && d.length ? sum / d.length : 0
};
"""


def _render_and_read():
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.get_screenshot(True)

    return pytest.headless.browser.execute_script(GRID)


def test_promoted_volume_gets_its_own_majorant():
    prepare()
    plot = pytest.plot

    n = 32
    grid = (np.mgrid[0:n, 0:n, 0:n] + 0.5) / n - 0.5
    radius = np.sqrt((grid ** 2).sum(axis=0))
    shell = np.where((radius > 0.30) & (radius < 0.34), 0.25, 0.0).astype(np.float32)
    ball = np.where(radius < 0.45, 1.0, 0.0).astype(np.float32)

    # identical everywhere the cache key used to look, which is the point
    common = {
        'color_map': [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'opacity_function': [0.0, 0.0, 1.0, 1.0],
        'color_range': [0.0, 1.0],
        'alpha_coef': 30.0,
        'bounds': [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
    }
    thin = k3d.volume(shell, **common)
    dense = k3d.volume(ball, **common)

    plot += thin
    plot += dense
    plot.renderer = "cinematic"
    plot.cinematic_samples = 1

    first = _render_and_read()
    assert first["cells"] > 0, "no majorant grid was built for the first volume"

    plot -= thin
    second = _render_and_read()

    assert second["traced"] != first["traced"], (
        "removing the first volume did not promote the second, so the swap was never exercised")

    # the ball fills the box and the shell barely touches it: no bound can serve both
    assert second["empty"] < first["empty"], (
        "the promoted volume kept the removed one's empty cells (%d, was %d): the grid describes "
        "a volume that is no longer being tracked, so its bound is below the extinction"
        % (second["empty"], first["empty"]))
    assert second["mean"] > 4.0 * first["mean"], (
        "the promoted volume's mean bound is %.3f against the removed one's %.3f: the grid did not "
        "follow the volume" % (second["mean"], first["mean"]))
