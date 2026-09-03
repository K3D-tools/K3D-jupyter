"""cinematic_seed decides whether the path tracer's noise is pinned.

The visual suite pins it (conftest passes cinematic_seed=1) so references compare bit for bit.
A notebook does not: by default every accumulation draws fresh noise, so two renders of the
same scene differ in their grain. This test reads the tracer's own flag through the diagnostic
handle rather than inferring it from pixels, then confirms the pixel-level consequence once.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare

STABLE_NOISE = "return window.__k3dTracer ? window.__k3dTracer.stableNoise : null"


def _render():
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()

    return Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGBA").tobytes()


def test_seed_pins_or_frees_the_noise():
    prepare()
    plot = pytest.plot
    plot += k3d.points(np.array([[0, 0, 0], [1, 1, 1], [-1, 1, 0]], dtype=np.float32),
                       point_size=0.6, shader="mesh", color=0x4477AA)
    plot.renderer = "cinematic"
    plot.cinematic_samples = 1

    try:
        # unpinned: the library's own Math.random paths, and the next accumulation looks different
        plot.cinematic_seed = None
        first = _render()
        assert pytest.headless.browser.execute_script(STABLE_NOISE) is False
        assert _render() != first

        # pinned: the same scene renders the same bytes, which is what the reference images rely on
        plot.cinematic_seed = 1
        first = _render()
        assert pytest.headless.browser.execute_script(STABLE_NOISE) is True
        assert _render() == first

        # the value travels through the headless diff, not just the trait
        assert plot.get_plot_params()["cinematicSeed"] == 1
    finally:
        plot.cinematic_seed = 1
        plot.cinematic_samples = 64
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)
