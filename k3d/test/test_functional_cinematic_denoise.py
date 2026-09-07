"""cinematic_denoise is off at zero, and off means the traced image is untouched.

Zero is not a small amount of filtering, it is none: the parameter is the switch as well as the
strength, the way cinematic_bokeh_size is, so a plot nobody asked to filter has to render exactly
what it traced. Every cinematic reference image depends on that.

Above zero the filter has to actually run, and the cheapest statement of "it ran" that does not
rest on eyeballing a picture is that the image changed and got smoother while the accumulation
behind it stayed the same - same seed, same budget, same scene.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from traitlets import TraitError

import k3d

from .plot_compare import prepare

VERTICES = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                     [0, 0, 1.2]], dtype=np.float32)
INDICES = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4]], dtype=np.uint32)


def _render():
    pytest.headless.sync(hold_until_refreshed=True)

    png = pytest.headless.get_screenshot(True)

    return np.asarray(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float64)


def _roughness(image, mask):
    """Mean step between horizontal neighbours - grain raises it, smoothing lowers it."""
    steps = np.abs(np.diff(image.mean(axis=2), axis=1))

    return steps[mask[:, :-1] & mask[:, 1:]].mean()


def test_zero_is_off_and_above_zero_filters():
    prepare()
    plot = pytest.plot
    plot += k3d.mesh(VERTICES, INDICES, color=0x3F6BFA, roughness=0.4)
    plot.renderer = "cinematic"
    plot.cinematic_seed = 1
    plot.cinematic_samples = 16

    try:
        assert plot.cinematic_denoise == 0.0, "the default has to be off"

        raw = _render()
        again = _render()

        assert np.array_equal(raw, again), (
            "two renders of a pinned seed already differ, so this test cannot say anything "
            "about what the filter does")

        plot.cinematic_denoise = 2.0
        filtered = _render()

        assert not np.array_equal(raw, filtered), (
            "cinematic_denoise = 2 changed nothing: the parameter is registered but nothing "
            "consumes it, which is how a trait ends up silently dead")

        mask = (raw.sum(axis=2) > 0) | (filtered.sum(axis=2) > 0)

        assert _roughness(filtered, mask) < _roughness(raw, mask), (
            "the filtered image is not smoother than the raw one (%.3f against %.3f): it "
            "changed the picture without removing grain"
            % (_roughness(filtered, mask), _roughness(raw, mask)))

        # and back to zero puts the traced image back exactly, not approximately
        plot.cinematic_denoise = 0.0

        assert np.array_equal(_render(), raw), (
            "returning cinematic_denoise to zero did not restore the traced image, so zero is "
            "not off and every cinematic reference depends on this parameter")

        # the value has to travel through the headless diff, not just the trait
        plot.cinematic_denoise = 1.5
        assert plot.get_plot_params()["cinematicDenoise"] == 1.5
    finally:
        plot.cinematic_denoise = 0.0
        plot.cinematic_samples = 64
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)


def test_a_negative_strength_is_refused():
    # the trait validator, so the refusal happens before a browser is involved
    with pytest.raises(TraitError):
        k3d.plot(cinematic_denoise=-1.0)
