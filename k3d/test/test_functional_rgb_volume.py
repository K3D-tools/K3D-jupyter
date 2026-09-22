"""A volume whose last axis is 3 or 4 carries colour per voxel, and the colour is the measurement.

These ask questions whose answers are known before rendering. The renderer writes without a
colour-space conversion (outputColorSpace is LinearSRGBColorSpace), textures carry no declared
colour space and tone mapping is off in the harness, so a slice of an RGB volume has to come back
out of the framebuffer as the exact bytes that went in. That is a fact a reference image cannot
check: a picture in roughly the right colours passes a comparison as soon as someone accepts it
once.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare

# four flat quadrants, chosen so no two share a channel value
QUADRANTS = {
    (0, 0): (255, 0, 0),
    (0, 1): (0, 255, 0),
    (1, 0): (0, 0, 255),
    (1, 1): (255, 255, 0),
}


def _quadrant_volume(n=8):
    """(n, n, n, 3) uint8, one flat colour per quadrant of the xy plane."""
    volume = np.zeros((n, n, n, 3), np.uint8)

    for (qy, qx), colour in QUADRANTS.items():
        ys = slice(qy * n // 2, (qy + 1) * n // 2)
        xs = slice(qx * n // 2, (qx + 1) * n // 2)
        volume[:, ys, xs] = colour

    return volume


def _screenshot():
    pytest.headless.sync(hold_until_refreshed=True)

    return np.asarray(Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGB"))


def _patch(image, fx, fy, half=6):
    """Median of a small window at (fx, fy) in image fractions - robust against edge pixels."""
    y = int(image.shape[0] * fy)
    x = int(image.shape[1] * fx)
    window = image[y - half:y + half, x - half:x + half].reshape(-1, 3)

    return np.median(window, axis=0).astype(int)


def test_rgb_volume_slice_shows_the_bytes_it_was_given():
    """Each quadrant of the slice is the colour stored in it, to the byte."""
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera_auto_fit = False
    pytest.plot.background_color = 0x000000

    # nearest sampling: linear would blend the quadrant borders, and the claim is about texels
    pytest.plot += k3d.volume_slice(
        _quadrant_volume(), slice_x=-1, slice_y=-1, slice_z=4, interpolation=0
    )
    # straight down -z, close enough that the quad covers the middle of the canvas
    pytest.plot.camera = [0, 0, 1.15, 0, 0, 0, 0, 1, 0]

    image = _screenshot()

    # the quad spans [-0.5, 0.5]; sample well inside each quadrant
    seen = {
        (0, 0): _patch(image, 0.35, 0.65),
        (0, 1): _patch(image, 0.65, 0.65),
        (1, 0): _patch(image, 0.35, 0.35),
        (1, 1): _patch(image, 0.65, 0.35),
    }

    for key, expected in QUADRANTS.items():
        assert tuple(seen[key]) == expected, (
            "quadrant %s came back %s, expected %s - the slice is not showing the data"
            % (key, tuple(seen[key]), expected)
        )


@pytest.mark.parametrize("colour,channel", [(0xff0000, 0), (0x00ff00, 1), (0x0000ff, 2)])
def test_rgb_volume_takes_its_hue_from_the_data(colour, channel):
    """A uniform RGB volume renders in its own colour; a colormap would answer the same for all three."""
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera_auto_fit = False
    pytest.plot.background_color = 0x000000

    rgb = np.zeros((8, 8, 8, 3), np.uint8)
    rgb[..., channel] = 255

    pytest.plot += k3d.volume(
        rgb, opacity_function=[0.0, 1.0, 1.0, 1.0], alpha_coef=200.0, samples=256.0
    )
    pytest.plot.camera = [0, 0, 1.6, 0, 0, 0, 0, 1, 0]

    middle = _patch(_screenshot(), 0.5, 0.5)
    others = [middle[i] for i in range(3) if i != channel]

    assert middle[channel] > 40, "the volume did not render: %s" % (tuple(middle),)
    assert middle[channel] > 3 * max(max(others), 1), (
        "channel %d should dominate, got %s" % (channel, tuple(middle))
    )


def test_mip_returns_the_colour_of_the_brightest_voxel():
    """The maximum is taken over luminance, and the colour comes from the voxel that won it.

    The filler is blue and the winning slab is orange - two colours no colormap over this data
    could produce, and the one that wins is not the one in front of the camera.
    """
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera_auto_fit = False
    pytest.plot.background_color = 0x000000

    rgb = np.zeros((16, 16, 16, 3), np.uint8)
    rgb[..., 2] = 200                       # blue everywhere, luminance 0.06
    rgb[6:10] = (255, 128, 0)               # an orange slab in the middle, luminance 0.57

    pytest.plot += k3d.mip(rgb, samples=256.0)
    pytest.plot.camera = [0, 0, 1.6, 0, 0, 0, 0, 1, 0]

    r, g, b = _patch(_screenshot(), 0.5, 0.5)

    assert r > 40, "the mip did not render: %s" % ((r, g, b),)
    assert r > g > b, "expected the orange slab, got %s" % ((r, g, b),)
    assert b < r // 3, "the blue filler is still dominating: %s" % ((r, g, b),)


def test_cinematic_leaves_an_rgb_volume_to_the_raster_layer():
    """The tracer has no medium for colour per voxel, and must not trace it as if it had.

    Its medium reads density from the red channel, so a pure green volume is the sharp case:
    traced, it would be empty; left to the raster layer, it is green.
    """
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera_auto_fit = False
    pytest.plot.background_color = 0x000000

    rgb = np.zeros((8, 8, 8, 3), np.uint8)
    rgb[..., 1] = 255

    pytest.plot += k3d.volume(
        rgb, opacity_function=[0.0, 1.0, 1.0, 1.0], alpha_coef=200.0, samples=256.0
    )
    pytest.plot.camera = [0, 0, 1.6, 0, 0, 0, 0, 1, 0]
    pytest.plot.cinematic_samples = 8
    pytest.plot.renderer = "cinematic"

    r, g, b = _patch(_screenshot(), 0.5, 0.5)

    pytest.plot.renderer = "simple"

    assert g > 40, "nothing was drawn: %s - the volume was traced as red-channel density" % ((r, g, b),)
    assert g > 3 * max(r, b, 1), "expected green, got %s" % ((r, g, b),)


def test_opacity_function_gates_an_rgb_volume():
    """The alpha of an RGB volume comes from opacity_function over luminance, and nothing else can."""
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera_auto_fit = False
    pytest.plot.background_color = 0x000000

    rgb = np.full((8, 8, 8, 3), 200, np.uint8)

    volume = k3d.volume(
        rgb, opacity_function=[0.0, 1.0, 1.0, 1.0], alpha_coef=200.0, samples=256.0
    )
    pytest.plot += volume
    pytest.plot.camera = [0, 0, 1.6, 0, 0, 0, 0, 1, 0]

    opaque = _patch(_screenshot(), 0.5, 0.5)

    volume.opacity_function = [0.0, 0.0, 1.0, 0.0]
    transparent = _patch(_screenshot(), 0.5, 0.5)

    assert opaque.max() > 40, "the volume never appeared: %s" % (tuple(opaque),)
    assert transparent.max() < 8, (
        "a zero opacity_function still left %s on a black background" % (tuple(transparent),)
    )
