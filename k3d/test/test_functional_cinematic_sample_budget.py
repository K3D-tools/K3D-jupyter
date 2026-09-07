"""Every sample that is traced has to reach the image.

PathTracingRenderer accumulates into two blend targets, and renderTask swaps only the local
handles - `_blendTargets` itself is never reordered - while `get target()` hands back the fixed
slot `_blendTargets[1]`. Sample k lands in `_blendTargets[k % 2]`, so slot 1 holds a mean only
after an ODD number of samples. At the default budget of 64 the presented and screenshotted
image was therefore the 63-sample mean: the last sample of every even accumulation was traced,
blended, and then not looked at.

The cheapest statement of that is the smallest even budget. Two samples used to present the
one-sample mean, so budgets 1 and 2 rendered byte-identical images while one of them did twice
the work. Comparing against a converged render says the second half of it: two samples must sit
closer to convergence than one, which cannot hold while they are the same image.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare

VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                     [0, 0, 1.2]], dtype=np.float32)
INDICES = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4]], dtype=np.uint32)


def _render(plot, samples):
    plot.cinematic_samples = samples
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()

    png = pytest.headless.get_screenshot(True)

    return np.asarray(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float64)


def test_an_even_budget_shows_every_sample_it_traced():
    prepare()
    plot = pytest.plot
    plot += k3d.mesh(VERTICES, INDICES, color=0x3F6BFA)
    plot.renderer = "cinematic"
    plot.cinematic_seed = 1

    try:
        one = _render(plot, 1)
        two = _render(plot, 2)

        assert not np.array_equal(one, two), (
            "budgets 1 and 2 rendered the same image: the second sample was traced and blended "
            "into the target the presenter does not read, so half the work was discarded")

        # and it is the second sample specifically, not just any difference: adding it has to
        # move the image towards where the accumulation is going, not merely somewhere else
        converged = _render(plot, 16)
        d_one = float(np.linalg.norm(one - converged))
        d_two = float(np.linalg.norm(two - converged))

        assert d_two < d_one, (
            "two samples sit %.1f from the converged render against one sample's %.1f: the extra "
            "sample did not bring the image closer to convergence" % (d_two, d_one))
    finally:
        plot.cinematic_samples = 64
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)
