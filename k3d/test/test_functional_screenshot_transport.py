"""The screenshot comes back over the session's HTTP channel, not as a script return value.

Every value returned from execute_script stays in the page's V8 heap for the life of the browser,
and a 4K PNG is about 10 MB, so a long animation reaches the tab's heap limit. get_screenshot
posts the image back instead. That is a transport nobody looks at until it breaks, and it breaks
silently - a truncated body or a race between the POST and the script resolving both produce
something that is not the picture - so these read the bytes rather than trusting the call.
"""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare

FRAMES = 6


def _plot():
    prepare()

    pytest.plot += k3d.points([0, 0, 0, 1, 1, 1, -1, 1, 0, 1, -1, 0],
                              point_size=0.4,
                              colors=[0xff0000, 0x00ff00, 0x0000ff, 0xffff00])
    pytest.plot.camera = [3, -3, 2, 0, 0, 0, 0, 0, 1]
    pytest.headless.sync(hold_until_refreshed=True)


def test_screenshot_arrives_as_a_png_of_the_right_size():
    """The bytes are a real PNG at the plot's own resolution, not an empty or partial body."""
    _plot()

    png = pytest.headless.get_screenshot(True)

    assert len(png) > 1000, "the body is too small to be an image"
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "the body does not start with a PNG signature"

    image = Image.open(BytesIO(png))

    assert image.format == "PNG"

    world = pytest.headless.browser.execute_script(
        "return [K3DInstance.getWorld().width, K3DInstance.getWorld().height];"
    )
    scale = pytest.plot.screenshot_scale

    assert image.size == (int(world[0] * scale), int(world[1] * scale))

    # a screenshot of a scene with four coloured points cannot be one flat colour
    assert len(np.unique(np.asarray(image.convert("RGB")).reshape(-1, 3), axis=0)) > 2


def test_repeated_screenshots_do_not_bleed_into_each_other():
    """A shared buffer on the server side would leak one frame into the next, or stall.

    The scene does not move, so every frame must come back byte for byte the same. A transport
    that answered with the previous body - or with a half-written one - shows up here as a
    difference, and one that never answered shows up as a timeout.
    """
    _plot()

    shots = [pytest.headless.get_screenshot(True) for _ in range(FRAMES)]

    assert len({bytes(s) for s in shots}) == 1, "a still scene produced different bytes"
    assert all(len(s) == len(shots[0]) for s in shots)


def test_screenshots_do_not_accumulate_in_the_page():
    """The point of the transport: the page's heap must not grow with the frame count."""
    _plot()

    memory = pytest.headless.get_memory()

    if memory is None or not memory["precise"]:
        pytest.skip("the browser does not report memory precisely "
                    "(needs --enable-precise-memory-info)")

    pytest.headless.get_screenshot(True)  # warm up outside the measurement
    before = pytest.headless.get_memory()["used_mb"]

    for _ in range(FRAMES):
        pytest.headless.get_screenshot(True)

    grew = pytest.headless.get_memory()["used_mb"] - before
    one = len(pytest.headless.get_screenshot(True)) / 1048576.0

    # generous: the old transport kept a full base64 payload per frame, so the failure this
    # guards against is FRAMES * 4/3 of an image, not a fraction of one
    assert grew < max(1.0, one * FRAMES * 0.5), (
        "the page kept %.2f MB over %d screenshots of %.2f MB each" % (grew, FRAMES, one)
    )
