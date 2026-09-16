"""A float16 time series has to blend by value, not by bit pattern.

The Float16Array stand-in is a Uint16Array of half-float bits, so the plain lerp the interpolator
used for every other dtype produced garbage between keyframes - 100 and 1000 met halfway at 322,
and a sign change went through the inf/NaN range. The same two frames in float32 are the oracle:
at t = 0.5 both volumes have to render the same picture.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

import k3d

from .plot_compare import prepare

SHAPE = (8, 8, 8)


def _volume(dtype):
    return k3d.volume(
        {
            "0.0": np.full(SHAPE, 100, dtype=dtype),
            "1.0": np.full(SHAPE, 1000, dtype=dtype),
        },
        color_range=[0, 1000],
        alpha_coef=50,
    )


def _shot(obj, time):
    prepare()
    pytest.plot += obj
    # the object first, the time after: set in the same sync that brings the object in, a time
    # is applied to an empty scene (headless.html applies plot_diff before objects_diff)
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.plot.time = time
    pytest.headless.sync(hold_until_refreshed=True)

    return Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGBA")


def _mismatch(a, b):
    return pixelmatch(a, b, Image.new("RGBA", a.size), threshold=0.1, includeAA=True)


def test_float16_frames_blend_like_float32():
    half = _shot(_volume(np.float16), 0.5)
    single = _shot(_volume(np.float32), 0.5)
    start = _shot(_volume(np.float32), 0.0)

    pixels = half.size[0] * half.size[1]

    # the midpoint is not the first keyframe - the interpolation did happen
    assert _mismatch(single, start) > 0.01 * pixels
    # and float16 lands where float32 does
    assert _mismatch(half, single) < 0.001 * pixels
