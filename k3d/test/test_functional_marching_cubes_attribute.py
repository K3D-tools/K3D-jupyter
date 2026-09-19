"""The marching cubes attribute has to be read where the surface is, on the grid it was given on.

The attribute is a 3D field sampled on the same grid as scalar_field, and the shader looks it up
per fragment at the fragment's own position in that grid. Nothing in a reference image says whether
the lookup lands in the right place - a wrong one still produces a coloured surface - so this asks
the question an image cannot, by making the answer computable in advance.

The scene is a sphere of a known radius in the middle of the field, and the attribute is a linear
ramp along one grid axis, shown through a colormap that runs blue at 0 to red at 1. Then the colour
at a pixel is known before rendering: a point on the silhouette circle at the fraction f across it
sits at grid coordinate (1 + RADIUS * (2f - 1)) / 2 along the ramp axis, so that is where in the
colormap it has to be. The test reads three points across the sphere and compares.

That pins the lookup from three sides at once. A coordinate that saturates - position + 0.5 over
geometry spanning 0 to the field size sends four vertices in five to the far face of the texture -
flattens the slope. A coordinate mapped to another axis puts the slope on the wrong screen axis. A
mirrored coordinate gets the sign wrong. None of them can hit the predicted three values.

Measured against the code as it was: every ramp read 0.86 where the grid puts 0.345, and the ball
read 0.887 on the half of it nearest the camera, which is past the equator. The flat attribute took
the browser to GL_INVALID_VALUE out of glTexStorage3D, the 1 x 1 x N texture over the size limit.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import k3d

from .plot_compare import prepare

N = 48
RADIUS = 0.62

# blue at 0, red at 1: two channels, so the position in the colormap survives being shaded -
# diffuse lighting scales both by the same factor and the ratio between them does not move
COLOR_MAP = [0.0, 0.0, 0.0, 1.0,
             1.0, 1.0, 0.0, 0.0]

# looking along +y with z up, so the screen runs +x to the right and +z upward
CAMERA = [0.0, -2.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# where across the silhouette the surface is read; kept off the rim, which is edge-on and dark
FRACTIONS = (0.25, 0.5, 0.75)
TOLERANCE = 0.06


def _expected(fraction):
    """Grid coordinate, and so the position in the colormap, at a fraction across the sphere."""
    return (1.0 + RADIUS * (2.0 * fraction - 1.0)) / 2.0


def _sphere_field():
    """A sphere of radius RADIUS about the middle of the box; the isosurface is at level 0.

    Negative inside, which is the sign that makes the triangles face outwards. The other sign
    builds the same sphere inside out, and back-face culling then shows its far half - measured:
    the visible surface reads the far hemisphere's attribute instead of the near one.
    """
    axis = np.linspace(-1.0, 1.0, N, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")

    return (np.sqrt(x * x + y * y + z * z) - RADIUS).astype(np.float32)


def _ramp(axis_name):
    """0 at one end of the named grid axis and 1 at the other, constant on the other two.

    The field is indexed [z][y][x], the same as scalar_field.
    """
    ramp = np.linspace(0.0, 1.0, N, dtype=np.float32)
    shape = {"x": (1, 1, N), "y": (1, N, 1), "z": (N, 1, 1)}[axis_name]

    return np.broadcast_to(ramp.reshape(shape), (N, N, N)).astype(np.float32).copy()


def _shot(attribute):
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera = list(CAMERA)
    pytest.plot += k3d.marching_cubes(
        _sphere_field(),
        level=0.0,
        attribute=attribute,
        color_range=[0.0, 1.0],
        color_map=COLOR_MAP,
        # a matte surface: a specular highlight is white and would pull the two channels together
        roughness=1.0,
        metalness=0.0,
        flat_shading=False,
    )
    pytest.headless.sync(hold_until_refreshed=True)

    return Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGB")


def _positions(img):
    """Per-pixel position in the colormap, and a mask of the pixels showing the surface.

    red / (red + blue) is 0 at the blue end of the map and 1 at the red end, and is unchanged by
    any lighting that scales the whole colour.
    """
    rgb = np.asarray(img, dtype=np.float64)
    height, width, _ = rgb.shape

    # the axes helper sits in a corner and is the only other coloured thing on the canvas
    inside = np.zeros((height, width), dtype=bool)
    inside[: int(height * 0.85), : int(width * 0.85)] = True

    red, blue = rgb[..., 0], rgb[..., 2]
    total = red + blue
    lit = total > 40.0
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > 25.0

    with np.errstate(invalid="ignore", divide="ignore"):
        position = np.where(total > 0.0, red / total, np.nan)

    return position, coloured & inside & lit


def _extent(mask, axis):
    """First and last row (axis 0) or column (axis 1) the surface covers."""
    used = np.nonzero(mask.any(axis=1 - axis))[0]
    assert used.size > 40, "the surface is not on the canvas"

    return int(used[0]), int(used[-1])


def _read(position, mask, axis, fraction):
    """Median colormap position in a narrow band at `fraction` across the surface along `axis`.

    The band is taken from the middle half of the other axis, where the sphere faces the camera
    and its colour is not being read at a grazing angle.
    """
    low, high = _extent(mask, axis)
    span = high - low
    centre = int(round(low + fraction * span))
    half = max(2, int(round(0.02 * span)))

    other_low, other_high = _extent(mask, 1 - axis)
    quarter = (other_high - other_low) // 4
    keep = slice(other_low + quarter, other_high - quarter + 1)

    if axis == 1:
        window = (keep, slice(centre - half, centre + half + 1))
    else:
        window = (slice(centre - half, centre + half + 1), keep)

    values = position[window][mask[window]]
    values = values[~np.isnan(values)]

    assert values.size > 10, f"no surface to read at {fraction} across axis {axis}"

    return float(np.median(values))


def _check_ramp(axis_name, screen_axis, flipped):
    img = _shot(_ramp(axis_name))
    position, mask = _positions(img)

    for fraction in FRACTIONS:
        # the screen's vertical axis counts downwards, so +z runs the other way across it
        read_at = 1.0 - fraction if flipped else fraction
        measured = _read(position, mask, screen_axis, read_at)
        expected = _expected(fraction)

        assert abs(measured - expected) < TOLERANCE, (
            f"the {axis_name} ramp reads {measured:.3f} at {fraction:.2f} across the sphere, "
            f"where the grid puts {expected:.3f}"
        )

    # and nothing of it leans along the other screen axis
    other = [_read(position, mask, 1 - screen_axis, f) for f in FRACTIONS]

    assert max(other) - min(other) < 2 * TOLERANCE, (
        f"the {axis_name} ramp also runs along the other screen axis ({other}): the attribute is "
        "being read along the wrong axis of the field"
    )


def test_a_ramp_along_x_runs_across_the_screen():
    _check_ramp("x", screen_axis=1, flipped=False)


def test_a_ramp_along_z_runs_up_the_screen():
    _check_ramp("z", screen_axis=0, flipped=True)


def test_a_ramp_along_the_view_direction_stays_on_the_near_half():
    """y points away from the camera, so the visible surface is the near hemisphere of the ball.

    Its grid coordinate runs from (1 - RADIUS) / 2 at the middle of the disc, the near pole, to
    1 / 2 at the rim, the equator - and no further. A mirrored lookup lands on the other side of
    the equator, on every pixel.
    """
    img = _shot(_ramp("y"))
    position, mask = _positions(img)

    values = position[mask]
    values = values[~np.isnan(values)]
    low, high = np.percentile(values, [5, 95])

    near_pole = (1.0 - RADIUS) / 2.0

    assert low > near_pole - TOLERANCE, (
        f"the surface reads {low:.3f} where the near pole of the ball is at {near_pole:.3f}"
    )
    assert high < 0.5 + TOLERANCE, (
        f"the surface reads {high:.3f}, past the equator at 0.500: the near half of the ball is "
        "being coloured with the far half's attribute"
    )

    # area-weighted, half the disc lies beyond radius 1/sqrt(2), which puts the median here
    expected = (1.0 - RADIUS * np.sqrt(0.5)) / 2.0
    measured = float(np.median(values))

    assert abs(measured - expected) < 2 * TOLERANCE, (
        f"the near half reads {measured:.3f} against the {expected:.3f} its geometry gives"
    )

    # and it leans on neither screen axis, which is what a swap of two axes would look like
    for axis, name in ((1, "across"), (0, "up")):
        band = [_read(position, mask, axis, f) for f in FRACTIONS]

        assert max(band) - min(band) < 2 * TOLERANCE, (
            f"a ramp along the view direction runs {name} the screen ({band}): two axes of the "
            "attribute are swapped"
        )


def test_a_flat_attribute_is_refused_rather_than_drawn_wrongly():
    """One value per vertex cannot be sampled at a position; it used to become a 1 x 1 x N texture."""
    prepare()

    pytest.plot.grid_visible = False
    pytest.plot.camera = list(CAMERA)
    pytest.plot += k3d.marching_cubes(
        _sphere_field(),
        level=0.0,
        attribute=np.linspace(0.0, 1.0, N * N * N, dtype=np.float32),
        color_range=[0.0, 1.0],
        color_map=COLOR_MAP,
        roughness=1.0,
        metalness=0.0,
    )
    pytest.headless.sync(hold_until_refreshed=True)

    img = Image.open(BytesIO(pytest.headless.get_screenshot(True))).convert("RGB")
    position, mask = _positions(img)
    read = [_read(position, mask, 1, f) for f in FRACTIONS]

    assert max(read) - min(read) < TOLERANCE, (
        f"a flat attribute still shades the surface through the colormap ({read}): it is being "
        "sampled as if it were a field"
    )

    messages = " | ".join(entry.get("message", "")
                          for entry in pytest.headless.browser.get_log("browser"))

    assert "attribute is sampled as a 3D field" in messages, (
        "the attribute was dropped without saying so"
    )
