import numpy as np
import pytest

import k3d

from .plot_compare import compare, prepare


def test_surface():
    prepare()

    Nx = 240
    Ny = 480
    xmin, xmax = -3, 3
    ymin, ymax = -0, 3

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    x, y = np.meshgrid(x, y)

    heights = np.sin(x ** 2 + y ** 2).astype(np.float32)

    surface = k3d.surface(heights, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    pytest.plot += surface

    compare("surface")

    surface.color = 0x00FFFF

    compare("surface_dynamic_color")

    surface.roughness = 0.06

    compare("surface_dynamic_roughness")


def test_surface_attribute():
    prepare()

    Nx = 240
    Ny = 480
    xmin, xmax = -3, 3
    ymin, ymax = -0, 3

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    x, y = np.meshgrid(x, y)

    heights = np.sin(x ** 2 + y ** 2).astype(np.float32)

    surface = k3d.surface(
        heights,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        attribute=heights,
        transform=k3d.transform(rotation=[np.pi / 4, 1, 0, 0]),
    )

    pytest.plot += surface

    compare("surface_attribute")


def test_surface_attribute_low():
    prepare()

    Nx = 24
    Ny = 48
    xmin, xmax = -3, 3
    ymin, ymax = -0, 3

    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    x, y = np.meshgrid(x, y)

    heights = np.sin(x ** 2 + y ** 2).astype(np.float32)

    surface = k3d.surface(
        heights,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        attribute=heights,
        transform=k3d.transform(rotation=[np.pi / 4, 1, 0, 0]),
    )

    pytest.plot += surface

    compare("surface_attribute_low")

    surface.flat_shading = False

    compare("surface_attribute_low_dynamic_smooth")

    surface.wireframe = True

    compare("surface_attribute_low_dynamic_wireframe")


def test_surface_heights_morph():
    """A same-shape heights update morphs the grid in place - the scene object
    survives (no delete/create round trip)."""
    prepare()

    Nx, Ny = 240, 480
    x = np.linspace(-3, 3, Nx)
    y = np.linspace(0, 3, Ny)
    x, y = np.meshgrid(x, y)

    surface = k3d.surface(
        np.sin(x ** 2 + y ** 2).astype(np.float32), xmin=-3, xmax=3, ymin=0, ymax=3
    )
    pytest.plot += surface
    pytest.headless.sync(hold_until_refreshed=True)

    uuid_before = pytest.headless.browser.execute_script(
        "return K3DInstance.getWorld().ObjectsById[%d].uuid;" % surface.id
    )

    surface.heights = np.cos(x * 3).astype(np.float32)

    compare("surface_heights_morph", modes=("simple",))

    uuid_after = pytest.headless.browser.execute_script(
        "return K3DInstance.getWorld().ObjectsById[%d].uuid;" % surface.id
    )
    assert uuid_before == uuid_after
