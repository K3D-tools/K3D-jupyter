import json
import os
from io import BytesIO

import pytest
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

from k3d.plot.plot_snapshot import _PLOT_PARAMS

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCES_DIR = os.path.join(TEST_DIR, "references")
RESULTS_DIR = os.path.join(TEST_DIR, "results")

# Cinematic references are half scale (640x360) at this budget; changing either invalidates
# them - every file under references/cinematic has to be regenerated.
#
# Why a budget this low is enough to catch regressions: cinematic_seed is pinned in
# conftest, so a render is bit-reproducible, and the comparison below allows zero
# mismatched pixels. Any change that moves the image fails the test at 16 samples exactly
# as it would at 256. Sample count buys convergence, and convergence is not what a
# reference comparison measures.
#
# Why not lower still: a reference is also read by a person. When a test fails, the diff
# has to let someone tell a darker material from a re-rolled noise field, and that is what
# stops being legible first.
REF_SAMPLES = 16
CINEMATIC_SCREENSHOT_SCALE = 0.5

# Glyph edges move by a few pixels between freetype versions (Debian image vs CI runner); a
# misplaced or missing label differs by hundreds, so text tests tolerate this much and no more.
GLYPH_AA_BUDGET = 32

# Modes listed in K3D_ACCEPT_REFERENCES ("cinematic", "simple,advanced", "all") have their
# renders written as the new reference instead of asserted. Never set in CI.
ACCEPT_REFERENCES = [
    mode.strip()
    for mode in os.environ.get("K3D_ACCEPT_REFERENCES", "").split(",")
    if mode.strip()
]

# Every reference this run overwrote. conftest turns a non-empty list into a non-zero exit:
# an accepting run asserts nothing, so it must never be readable as a passing one.
ACCEPTED = []

# Every plot parameter as the harness plot was born, captured by conftest before the first test.
# prepare() restores all of them, so a trait added to the plot is covered the day it is added
# rather than the day someone remembers to extend a list here.
BASELINE = {}

# Restored in the page instead of on the plot, or not restorable from a value at all.
#   mode, camera: a change made in the browser never reaches the plot, so assigning the same
#     value produces no diff and never arrives
#   depth_peels: prepare() takes it as an argument
_BASELINE_SKIP = {"mode", "camera", "depthPeels"}


# What drew the committed references. A reference is only ground truth for the browser and
# rasterizer that made it: the Dockerfile pins Chrome for exactly this reason, and a run in a
# different one produces a wall of pixel differences that says nothing about the change under test.
ENVIRONMENT_PATH = os.path.join(REFERENCES_DIR, "ENVIRONMENT.json")

# Mismatches found at session start, reported once at the end rather than per test.
ENVIRONMENT_MISMATCH = []


def _environment(headless):
    """Browser and rasterizer identity, as the references record it."""
    info = headless.get_gl_info() or {}

    return {
        "browserVersion": headless.browser.capabilities.get("browserVersion"),
        "unmaskedRenderer": info.get("unmaskedRenderer"),
        "maxTextureSize": info.get("maxTextureSize"),
    }


def check_environment(headless):
    """Compare this run's renderer against the one the references were drawn with.

    Writes the file instead when the run is accepting references: whatever it draws becomes the
    new ground truth, so the environment that drew it is part of that record.
    """
    actual = _environment(headless)

    if ACCEPT_REFERENCES:
        with open(ENVIRONMENT_PATH, "w", encoding="utf-8") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")

        return actual

    if not os.path.isfile(ENVIRONMENT_PATH):
        return actual

    with open(ENVIRONMENT_PATH, encoding="utf-8") as f:
        expected = json.load(f)

    for key, want in expected.items():
        if actual.get(key) != want:
            ENVIRONMENT_MISMATCH.append((key, want, actual.get(key)))

    return actual


def capture_baseline(plot):
    """Record the plot's parameters as the state every test starts from."""
    BASELINE.clear()
    BASELINE.update(plot.get_plot_params())

    return BASELINE


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

    # Every parameter back to how the harness plot was born. The hand-written list this
    # replaced covered 26 of 65, and the 39 it missed leaked between tests - slice_viewer_object_id
    # pointed at an object prepare() had already removed for every test after the slice viewer ran.
    for key, trait in _PLOT_PARAMS:
        if key in _BASELINE_SKIP or key not in BASELINE:
            continue

        value = BASELINE[key]
        setattr(pytest.plot, trait, list(value) if isinstance(value, list) else value)

    pytest.plot.depth_peels = depth_peels
    pytest.plot.camera = [2, -3, 0.2, 0.0, 0.0, 0.0, 0, 0, 1]
    # and in the page, like mode: a camera moved in the browser (a drag, a manipulator) never
    # reaches the plot, so assigning the same value there produces no diff and does not arrive
    pytest.headless.browser.execute_script(
        "if (K3DInstance) { K3DInstance.setCamera(arguments[0]); }", pytest.plot.camera
    )
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()


def compare(
        name,
        only_canvas=True,
        threshold=0.2,
        max_mismatched_pixels=0,
        camera_factor=1.0,
        modes=("simple", "advanced", "cinematic"),
):
    """Compare the current plot against a stored reference image, in every renderer mode.

    Two independent knobs, previously conflated into one:

    threshold             per-pixel colour-distance tolerance passed to pixelmatch,
                          a fraction in 0..1. Governs when a single pixel counts as
                          different at all. pixelmatch calls a pixel different when the
                          YIQ distance exceeds 35215 * threshold^2, so the default 0.2
                          lets a uniform shift of 52 levels per channel through on every
                          pixel of the image. That tolerance is what absorbs driver-level
                          antialiasing differences; it is not an exact match, and a change
                          in exposure, tone mapping or light intensity can hide under it.
    max_mismatched_pixels how many differing pixels the image may still contain and
                          pass, as an absolute count (pixelmatch's return value).
                          0 means no pixel may differ *by more than threshold*.

    Note that pixelmatch returns a pixel count, so the two knobs are not interchangeable.
    Pass threshold=0 for a comparison that answers "did this image change at all".

    The advanced render is compared against references/advanced/<name>.png. When that file
    does not exist, it is compared against the simple reference: no file means "advanced has
    no right to change this image", which is how the contract for unlit scenes is enforced.
    Cinematic has no such fallback: a missing references/cinematic/<name>.png is a failure.
    """
    for mode in modes:
        if pytest.plot.renderer != mode:
            pytest.plot.renderer = mode

        if mode == "cinematic":
            pytest.plot.cinematic_samples = REF_SAMPLES
            pytest.plot.screenshot_scale = CINEMATIC_SCREENSHOT_SCALE

        try:
            pytest.headless.sync(hold_until_refreshed=True)

            if camera_factor is not None:
                pytest.headless.camera_reset(camera_factor)

            screenshot = pytest.headless.get_screenshot(only_canvas)
        finally:
            if mode == "cinematic":
                pytest.plot.screenshot_scale = 1.0

        result = Image.open(BytesIO(screenshot))
        img_diff = Image.new("RGBA", result.size)
        reference = None

        ref_name = name if mode == "simple" else mode + "/" + name
        reference_path = os.path.join(REFERENCES_DIR, ref_name + ".png")
        if mode == "advanced" and not os.path.isfile(reference_path):
            reference_path = os.path.join(REFERENCES_DIR, name + ".png")
        if os.path.isfile(reference_path):
            reference = Image.open(reference_path)

        if mode in ACCEPT_REFERENCES or "all" in ACCEPT_REFERENCES:
            accepted_path = os.path.join(REFERENCES_DIR, ref_name + ".png")

            # Write an advanced reference only when the render differs from the simple fallback.
            if (mode == "advanced" and not os.path.isfile(accepted_path)
                    and reference is not None and result.size == reference.size):
                unchanged = pixelmatch(result, reference,
                                       Image.new("RGBA", result.size),
                                       threshold=threshold, includeAA=True)
                if unchanged <= max_mismatched_pixels:
                    continue

            os.makedirs(os.path.dirname(accepted_path), exist_ok=True)
            result.save(accepted_path)
            ACCEPTED.append(ref_name)
            print("accepted", ref_name)
            continue

        assert reference is not None, (
            "%s [%s]: no reference at %s. An empty image would pass for any white scene, which is "
            "what a test that rendered nothing produces - run with K3D_ACCEPT_REFERENCES to write "
            "one." % (name, mode, reference_path)
        )

        mismatch = pixelmatch(
            result, reference, img_diff, threshold=threshold, includeAA=True
        )

        if mismatch > max_mismatched_pixels:
            os.makedirs(os.path.join(RESULTS_DIR, mode), exist_ok=True)

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
