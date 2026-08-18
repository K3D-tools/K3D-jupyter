import os
import pytest
from PIL import Image
from io import BytesIO
from pixelmatch.contrib.PIL import pixelmatch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCES_DIR = os.path.join(TEST_DIR, "references")
RESULTS_DIR = os.path.join(TEST_DIR, "results")


def prepare(depth_peels=0):
    # mode is not a synced trait, so it can only be reset in the page. A plot left in manipulate
    # mode attaches a gizmo to every object of every later test.
    # Reset in the page, not through the plot: a change made in the browser is invisible to the
    # sync diff, so assigning the same value on the plot produces no diff and never arrives.
    pytest.headless.browser.execute_script(
        "if (K3DInstance) { K3DInstance.setViewMode('view'); K3DInstance.setTime(0); }"
    )

    while len(pytest.plot.objects) > 0:
        pytest.plot -= pytest.plot.objects[-1]

    pytest.plot.clipping_planes = []
    pytest.plot.colorbar_object_id = 0
    pytest.plot.grid_visible = True
    pytest.plot.depth_peels = depth_peels
    pytest.plot.rendering_steps = 1
    pytest.plot.renderer = "simple"
    pytest.plot.environment = "neutral"
    pytest.plot.environment_rotation = 0.0
    pytest.plot.tone_mapping = "none"
    pytest.plot.ao_radius = 0.07
    pytest.plot.ao_strength = 1.8
    pytest.plot.camera_mode = "trackball"
    pytest.plot.camera = [2, -3, 0.2, 0.0, 0.0, 0.0, 0, 0, 1]
    pytest.plot.background_color = 0xFFFFFF
    pytest.plot.camera_fov = 60.0
    pytest.plot.time = 0.0
    pytest.plot.time_interpolation = True
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()


def compare(
        name,
        only_canvas=True,
        threshold=0.2,
        max_mismatched_pixels=0,
        camera_factor=1.0,
        modes=("simple", "advanced"),
):
    """Compare the current plot against a stored reference image, in every renderer mode.

    Two independent knobs, previously conflated into one:

    threshold             per-pixel colour-distance tolerance passed to pixelmatch,
                          a fraction in 0..1. Governs when a single pixel counts as
                          different at all.
    max_mismatched_pixels how many differing pixels the image may still contain and
                          pass, as an absolute count (pixelmatch's return value).
                          0 keeps the historical behaviour of demanding an exact match.

    Note that pixelmatch returns a pixel count, so the two knobs are not interchangeable.

    The advanced render is compared against references/advanced/<name>.png. When that file
    does not exist, it is compared against the simple reference: no file means "advanced has
    no right to change this image", which is how the contract for unlit scenes is enforced.
    """
    for mode in modes:
        if pytest.plot.renderer != mode:
            pytest.plot.renderer = mode

        pytest.headless.sync(hold_until_refreshed=True)

        if camera_factor is not None:
            pytest.headless.camera_reset(camera_factor)

        screenshot = pytest.headless.get_screenshot(only_canvas)

        result = Image.open(BytesIO(screenshot))
        img_diff = Image.new("RGBA", result.size)
        reference = None

        ref_name = name if mode == "simple" else "advanced/" + name
        reference_path = os.path.join(REFERENCES_DIR, ref_name + ".png")
        if mode == "advanced" and not os.path.isfile(reference_path):
            reference_path = os.path.join(REFERENCES_DIR, name + ".png")
        if os.path.isfile(reference_path):
            reference = Image.open(reference_path)

        if reference is None:
            reference = Image.new("RGBA", result.size)

        mismatch = pixelmatch(
            result, reference, img_diff, threshold=threshold, includeAA=True
        )

        if mismatch > max_mismatched_pixels:
            os.makedirs(os.path.join(RESULTS_DIR, "advanced"), exist_ok=True)

            with open(os.path.join(RESULTS_DIR, ref_name + ".k3d"), "wb") as f:
                f.write(pytest.plot.get_binary_snapshot(1))
            result.save(os.path.join(RESULTS_DIR, ref_name + ".png"))
            reference.save(os.path.join(RESULTS_DIR, ref_name + "_reference.png"))
            img_diff.save(os.path.join(RESULTS_DIR, ref_name + "_diff.png"))

            print(ref_name, mismatch, max_mismatched_pixels)

        assert mismatch <= max_mismatched_pixels, (
            "%s [%s]: %d pixel(s) differ from the reference (budget %d, per-pixel threshold %g); "
            "artifacts written to %s"
            % (name, mode, mismatch, max_mismatched_pixels, threshold, RESULTS_DIR)
        )

    if len(modes) > 1 and pytest.plot.renderer != "simple":
        pytest.plot.renderer = "simple"
