"""cinematic_denoise is off at zero, and off means the traced image is untouched.

Zero is not a small amount of denoising, it is none: the parameter is the switch as well as the
strength, the way cinematic_bokeh_size is, so a plot nobody asked to denoise has to render exactly
what it traced. Every cinematic reference image depends on that.

Above zero Open Image Denoise has to actually run, and the cheapest statement of "it ran" that
does not rest on eyeballing a picture is that the image changed and got smoother while the
accumulation behind it stayed the same - same seed, same budget, same scene.

The screenshots are small on purpose: the suite's WebGPU is SwiftShader, on the CPU, where the
network costs seconds per 300 pixels square. 320 x 180 is also not square, which is the shape
oidn-web 0.4.0 reads past the end of unless the image is padded first.
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
    plot.screenshot_scale = 0.25

    try:
        assert plot.cinematic_denoise == 0.0, "the default has to be off"

        raw = _render()
        again = _render()

        assert np.array_equal(raw, again), (
            "two renders of a pinned seed already differ, so this test cannot say anything "
            "about what the filter does")

        plot.cinematic_denoise = 1.0
        filtered = _render()

        assert not np.array_equal(raw, filtered), (
            "cinematic_denoise = 1 changed nothing: the parameter is registered but nothing "
            "consumes it, or the browser has no WebGPU and the denoiser stepped aside")

        assert np.array_equal(_render(), filtered), (
            "two denoised renders of a pinned seed differ: the network is not deterministic "
            "here, so no reference image can be taken of a denoised plot")

        mask = (raw.sum(axis=2) > 0) | (filtered.sum(axis=2) > 0)

        assert _roughness(filtered, mask) < _roughness(raw, mask), (
            "the filtered image is not smoother than the raw one (%.3f against %.3f): it "
            "changed the picture without removing grain"
            % (_roughness(filtered, mask), _roughness(raw, mask)))

        # Denoised, not replaced: at large scale the image is still the traced one. A read past
        # the end of the image comes back NaN for every pixel, which is nothing like it.
        def coarse(image):
            h, w = image.shape[0] // 20 * 20, image.shape[1] // 20 * 20
            return image[:h, :w].reshape(h // 20, 20, w // 20, 20, 3).mean(axis=(1, 3))

        shift = np.abs(coarse(filtered) - coarse(raw)).mean()

        assert shift < 12.0, (
            "the denoised image is %.1f levels away from the traced one at large scale: it "
            "is not a denoised version of it" % shift)

        # between 0 and 1 the two are mixed, so a half lands between them
        plot.cinematic_denoise = 0.5
        half = _render()

        assert np.abs(half - raw).mean() < np.abs(filtered - raw).mean(), (
            "cinematic_denoise = 0.5 is no closer to the traced image than 1 is")

        # and back to zero puts the traced image back exactly, not approximately
        plot.cinematic_denoise = 0.0

        assert np.array_equal(_render(), raw), (
            "returning cinematic_denoise to zero did not restore the traced image, so zero is "
            "not off and every cinematic reference depends on this parameter")

        # the value has to travel through the headless diff, not just the trait
        plot.cinematic_denoise = 0.75
        assert plot.get_plot_params()["cinematicDenoise"] == 0.75
    finally:
        plot.cinematic_denoise = 0.0
        plot.screenshot_scale = 1.0
        plot.cinematic_samples = 64
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)


def test_a_negative_strength_is_refused():
    # the trait validator, so the refusal happens before a browser is involved
    with pytest.raises(TraitError):
        k3d.plot(cinematic_denoise=-1.0)


def test_a_volume_is_denoised_without_surface_buffers():
    # a medium has no first surface: the colour-only network, and coverage denoised as well
    prepare()
    plot = pytest.plot
    z, y, x = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
    blob = np.exp(-4 * (x ** 2 + y ** 2 + z ** 2)).astype(np.float32)
    plot += k3d.volume(blob, color_range=[0.1, 1.0], alpha_coef=20)
    plot.renderer = "cinematic"
    plot.cinematic_seed = 1
    plot.cinematic_samples = 8
    plot.screenshot_scale = 0.25

    try:
        raw = _render()
        plot.cinematic_denoise = 1.0
        filtered = _render()

        assert not np.array_equal(raw, filtered), "the volume was not denoised"
        assert np.array_equal(_render(), filtered), "the denoised volume is not deterministic"

        mask = (raw.sum(axis=2) > 0) | (filtered.sum(axis=2) > 0)

        assert _roughness(filtered, mask) < _roughness(raw, mask), (
            "the denoised volume is not smoother than the traced one")
    finally:
        plot.cinematic_denoise = 0.0
        plot.screenshot_scale = 1.0
        plot.cinematic_samples = 64
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)


def _checker_png(size, cells):
    y, x = np.mgrid[0:size, 0:size] * cells // size
    mask = ((x + y) % 2 == 0)[..., None]
    pixels = np.where(mask, np.array([240, 240, 240], np.uint8), np.array([20, 20, 20], np.uint8))
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _checker_quad(x0, x1, y, z0, z1, cells):
    vertices = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], np.float32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    return k3d.mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.uint32), uvs=uvs, side="double",
                    texture=_checker_png(512, cells))


def _fine(image, sigma=1.5):
    """The image less a gaussian blur of it: what is left is edges and grain."""
    grey = image.mean(axis=2)
    r = int(3 * sigma)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    padded = np.pad(grey, r, mode="edge")
    rows = sum(k[i] * padded[:, i:i + grey.shape[1]] for i in range(2 * r + 1))
    return grey - sum(k[i] * rows[i:i + grey.shape[0]] for i in range(2 * r + 1))


def test_depth_of_field_blurs_the_guides_as_well():
    """Out of focus, the denoised image must keep the blur, not the albedo guide's edges.

    OIDN keeps whatever edges its albedo and normal guides show. Rasterised through a pinhole,
    the guides showed a defocused checker sharp, and the network drew harder squares into the
    blur than the lens leaves there. The guides are now averaged over the lens like the trace.
    Measured against 1024 traced samples, the fine detail of the denoised background was 8.1
    levels off with sharp guides and is 5.1 with blurred ones.
    """
    prepare()
    plot = pytest.plot
    # in focus: a small checker in the middle; behind it, out of focus, a large one
    plot += _checker_quad(-0.6, 0.6, 0.0, -0.4, 0.4, 8)
    plot += _checker_quad(-12.0, 12.0, 12.0, -8.0, 8.0, 30)
    plot.renderer = "cinematic"
    plot.camera_auto_fit = False
    plot.camera = [0, -3, 0, 0, 0, 0, 0, 0, 1]
    plot.cinematic_focus_distance = 3.0
    plot.cinematic_bokeh_size = 0.1
    # 512 x 288, one OIDN tile on SwiftShader
    plot.screenshot_scale = 0.4

    try:
        plot.cinematic_seed = 7
        plot.cinematic_samples = 1024
        reference = _render()
        plot.cinematic_seed = 1
        plot.cinematic_samples = 32
        plot.cinematic_denoise = 1.0
        denoised = _render()

        h, w = reference.shape[:2]
        # a band of the defocused checker left of the near one
        background = np.zeros((h, w), bool)
        background[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.3)] = True
        off = np.sqrt(((_fine(denoised) - _fine(reference))[background] ** 2).mean())

        assert off < 6.5, (
            "out of focus, the fine detail of the denoised image is %.1f levels from the traced "
            "one (8.1 with guides rendered through a pinhole): the guides are not blurred by the "
            "lens" % off)
    finally:
        plot.cinematic_bokeh_size = 0.0
        plot.cinematic_focus_distance = 0.0
        plot.cinematic_denoise = 0.0
        plot.screenshot_scale = 1.0
        plot.cinematic_samples = 64
        plot.camera_auto_fit = True
        plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)
