import os
import pytest
from PIL import Image
from io import BytesIO
from pixelmatch.contrib.PIL import pixelmatch

# Resolve fixtures relative to this file, not the process CWD: CI happens to run
# `cd k3d && python -m pytest`, but from anywhere else the old "./test/..." paths made
# every reference silently read as missing and the artifact writes raise FileNotFoundError.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCES_DIR = os.path.join(TEST_DIR, "references")
RESULTS_DIR = os.path.join(TEST_DIR, "results")


def prepare(depth_peels=0):
    while len(pytest.plot.objects) > 0:
        pytest.plot -= pytest.plot.objects[-1]

    pytest.plot.clipping_planes = []
    pytest.plot.colorbar_object_id = 0
    pytest.plot.grid_visible = True
    pytest.plot.depth_peels = depth_peels
    pytest.plot.camera_mode = "trackball"
    pytest.plot.camera = [2, -3, 0.2, 0.0, 0.0, 0.0, 0, 0, 1]
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()


def compare(
        name,
        only_canvas=True,
        threshold=0.2,
        max_mismatched_pixels=0,
        camera_factor=1.0,
):
    """Compare the current plot against a stored reference image.

    Two independent knobs, previously conflated into one:

    threshold             per-pixel colour-distance tolerance passed to pixelmatch,
                          a fraction in 0..1. Governs when a single pixel counts as
                          different at all.
    max_mismatched_pixels how many differing pixels the image may still contain and
                          pass, as an absolute count (pixelmatch's return value).
                          0 keeps the historical behaviour of demanding an exact match.

    The old code asserted `mismatch < threshold`, i.e. compared a pixel *count* against
    0.2, so it only ever passed at exactly zero differing pixels and no caller could
    actually loosen it.
    """
    pytest.headless.sync(hold_until_refreshed=True)

    if camera_factor is not None:
        pytest.headless.camera_reset(camera_factor)

    screenshot = pytest.headless.get_screenshot(only_canvas)

    result = Image.open(BytesIO(screenshot))
    img_diff = Image.new("RGBA", result.size)
    reference = None

    reference_path = os.path.join(REFERENCES_DIR, name + ".png")
    if os.path.isfile(reference_path):
        reference = Image.open(reference_path)

    if reference is None:
        reference = Image.new("RGBA", result.size)

    mismatch = pixelmatch(
        result, reference, img_diff, threshold=threshold, includeAA=True
    )

    if mismatch > max_mismatched_pixels:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(os.path.join(RESULTS_DIR, name + ".k3d"), "wb") as f:
            f.write(pytest.plot.get_binary_snapshot(1))
        result.save(os.path.join(RESULTS_DIR, name + ".png"))
        reference.save(os.path.join(RESULTS_DIR, name + "_reference.png"))
        img_diff.save(os.path.join(RESULTS_DIR, name + "_diff.png"))

        print(name, mismatch, max_mismatched_pixels)

    assert mismatch <= max_mismatched_pixels, (
        "%s: %d pixel(s) differ from the reference (budget %d, per-pixel threshold %g); "
        "artifacts written to %s"
        % (name, mismatch, max_mismatched_pixels, threshold, RESULTS_DIR)
    )
