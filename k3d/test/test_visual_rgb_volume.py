"""Visual coverage for a volume that carries colour per voxel.

test_functional_rgb_volume.py checks invariants - the bytes of a slice, which channel dominates,
that the ramp gates the march. Those hold for many pictures. These pin the picture itself, in
every renderer, so a change that keeps the invariants and still moves the image is caught.

The field is a colour cube inside a sphere: red ramps along x, green along y, blue along z, and
everything outside the sphere is black. Any axis swap, saturating coordinate or channel mix-up
shows up as a differently coloured ball rather than as a subtle shift.

The ramps start well above zero so the corner of the cube facing the camera is not black, and
each channel still spans 115 levels - twice what the comparison threshold absorbs - so a swapped
or clamped axis fails rather than hides.

The march needs its opacity ramp to start high, and that is the one thing here worth knowing.
A volume is sampled trilinearly, so the ball has a soft rim where the texture fades from nothing
to the colour. A scalar field never shows this: whatever value the ray stops at, the colormap
turns it into a full-intensity colour. In an RGB volume the value *is* the colour, so a ray that
stops halfway up the rim paints a half-bright one. Measured on a white ball: a ramp rising from
0.02 renders it at 76 levels, the same ball with the ramp rising from 0.45 renders at 249, and
with interpolation off the low ramp renders at 255 - no rim to stop in. The foot is at 0.45 here
so the surface forms where the data is already itself.

Backgrounds are picked per test by measurement too. The maximum-intensity projection of this ball
is pale, so on the suite's white page it would sit a few levels from the background and under the
comparison threshold; it gets a black page. Colour itself is pinned by the slice, which carries no
lighting at all and is compared byte for byte in the functional tests.
"""
import numpy as np
import pytest

import k3d

from .plot_compare import compare, prepare

N = 64


def _colour_ball(n=N):
    g = np.linspace(0.0, 1.0, n, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")

    # the floor is high because the simple renderer lights a volume with ambient alone and
    # light_scale reaches only the cinematic path, so brightness has to come from the data
    rgb = 0.55 + 0.45 * np.stack([x, y, z], axis=-1)
    inside = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) < 0.35 ** 2

    return (rgb * inside[..., None] * 255).astype(np.uint8)


def _stage(background=None):
    prepare()

    pytest.plot.grid_visible = False

    if background is not None:
        pytest.plot.background_color = background


def test_rgb_volume_slice():
    _stage()

    pytest.plot += k3d.volume_slice(
        _colour_ball(), slice_x=N // 2, slice_y=N // 2, slice_z=N // 2
    )

    # cinematic does not draw a slice at all - user/cinematic.rst says so - and a blank page
    # is what a failed render also produces, so there is nothing there worth pinning
    compare("rgb_volume_slice", modes=("simple", "advanced"))


def test_rgb_volume_slice_viewer():
    """The slice viewer draws the same texture through its own controls and its own camera.

    Its three panels are rebuilt on every direction change, which is a different path into
    VolumeSlice than the plain plot, so the colour has to survive it as well. Cinematic is out
    for the same reason the scalar slice-viewer tests leave it out: this is a camera mode.
    """
    _stage()

    volume = k3d.volume_slice(
        _colour_ball(), slice_x=N // 2, slice_y=N // 2, slice_z=N // 2
    )

    pytest.plot.camera_mode = "slice_viewer"
    pytest.plot.camera = [1, 1, 1, 0, 0, 0, 0, 0, 1]  # to force camera sync
    pytest.plot += volume

    pytest.plot.slice_viewer_object_id = volume.id

    pytest.plot.slice_viewer_direction = "z"
    volume.slice_x, volume.slice_y, volume.slice_z = -1, -1, N // 2
    compare("rgb_volume_slice_viewer_z", modes=("simple", "advanced"))

    pytest.plot.slice_viewer_direction = "x"
    volume.slice_x, volume.slice_y, volume.slice_z = N // 2, -1, -1
    compare("rgb_volume_slice_viewer_x", modes=("simple", "advanced"))


def test_rgb_volume():
    _stage()

    pytest.plot += k3d.volume(
        _colour_ball(),
        opacity_function=[0.0, 0.0, 0.45, 0.0, 0.55, 1.0, 1.0, 1.0],
        alpha_coef=120.0,
        samples=256.0,
    )

    compare("rgb_volume")


def test_rgb_mip():
    # black: a maximum-intensity projection of this ball is pale, and on the suite's white page
    # it would sit a few levels from the background, under the comparison threshold
    _stage(background=0x000000)

    pytest.plot += k3d.mip(_colour_ball(), samples=256.0)

    compare("rgb_mip")
