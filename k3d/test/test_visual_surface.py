import numpy as np
import pytest

import k3d
from .plot_compare import prepare, compare


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

    compare('surface')

    surface.color = 0x00ffff

    compare('surface_dynamic_color')

    surface.shininess = 500.0

    compare('surface_dynamic_shininess')


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

    surface = k3d.surface(heights, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, attribute=heights,
                          transform=k3d.transform(rotation=[np.pi / 4, 1, 0, 0]))

    pytest.plot += surface

    compare('surface_attribute')


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

    surface = k3d.surface(heights, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                          attribute=heights, transform=k3d.transform(rotation=[np.pi / 4, 1, 0, 0]))

    pytest.plot += surface

    compare('surface_attribute_low')

    surface.flat_shading = False

    compare('surface_attribute_low_dynamic_smooth')

    surface.wireframe = True

    compare('surface_attribute_low_dynamic_wireframe')
