import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

T = 1.618033988749895
r = 4.77
zmin, zmax = -r, r
xmin, xmax = -r, r
ymin, ymax = -r, r
Nx, Ny, Nz = 77, 77, 77

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
z = np.linspace(zmin, zmax, Nz)
x, y, z = np.meshgrid(x, y, z, indexing="ij")
p = 2 - (
        np.cos(x + T * y)
        + np.cos(x - T * y)
        + np.cos(y + T * z)
        + np.cos(y - T * z)
        + np.cos(z - T * x)
        + np.cos(z + T * x)
).astype(np.float32)
a = (
        np.sin(x + T * y)
        + np.sin(x - T * y)
        + np.sin(y + T * z)
        + np.cos(y - T * z)
        + np.sin(z - T * x)
        + np.cos(z + T * x)
).astype(np.float32)


def test_marching_cubes():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0)

    pytest.plot += iso

    compare("marching_cubes")

    iso.level = 0.1

    compare("marching_cubes_dynamic_level")


def test_marching_cubes_smoothed():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0, flat_shading=False)

    pytest.plot += iso

    compare("marching_cubes_smoothed")

    iso.shininess = 500.0

    compare("marching_cubes_smoothed_dynamic_shininess")


def test_marching_cubes_wireframe():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0, wireframe=True)

    pytest.plot += iso

    compare("marching_cubes_wireframe")


def test_marching_cubes_non_uniformly_spaced():
    prepare()

    ax = np.logspace(0, np.log10(xmax - xmin + 1), Nx, endpoint=True) + xmin - 1
    ay = np.logspace(0, np.log10(ymax - ymin + 1), Nx, endpoint=True) + ymin - 1
    az = np.logspace(0, np.log10(zmax - zmin + 1), Nx, endpoint=True) + zmin - 1

    x, y, z = np.meshgrid(ax, ay, az, indexing="ij")
    p = 2 - (
            np.cos(x + T * y)
            + np.cos(x - T * y)
            + np.cos(y + T * z)
            + np.cos(y - T * z)
            + np.cos(z - T * x)
            + np.cos(z + T * x)
    ).astype(np.float32)

    iso = k3d.marching_cubes(
        p,
        level=0.0,
        color=1,
        spacings_x=(ax[1:] - ax[:-1]).astype(np.float32),
        spacings_y=(ay[1:] - ay[:-1]).astype(np.float32),
        spacings_z=(az[1:] - az[:-1]).astype(np.float32),
        bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
        opacity=0.15,
        wireframe=True,
    )
    pytest.plot += iso

    compare("marching_cubes_non_uniformly_spaced")


def test_marching_cubes_opacity_depth_peels():
    global p, a

    prepare(depth_peels=8)

    iso = k3d.marching_cubes(p, level=0.0, opacity=0.8, attribute=a,
                             color_map=k3d.matplotlib_color_maps.Inferno)

    pytest.plot += iso

    compare("test_marching_cubes_opacity_depth_peels")


def test_marching_cubes_opacity():
    global p, a

    prepare()

    iso = k3d.marching_cubes(p, level=0.0, opacity=0.8, attribute=a,
                             color_map=k3d.matplotlib_color_maps.Inferno)

    pytest.plot += iso

    compare("test_marching_cubes_opacity")


def test_marching_cubes_with_attribute():
    global p, a

    prepare()

    iso = k3d.marching_cubes(p, attribute=a, level=0.0)

    pytest.plot += iso

    compare("marching_cubes_with_attribute")


def test_marching_cubes_with_dynamic_attribute():
    global p, a

    prepare()

    iso = k3d.marching_cubes(p, level=0.0)

    pytest.plot += iso

    compare("marching_cubes")

    iso.attribute = a
    iso.color_map = k3d.matplotlib_color_maps.Inferno
    iso.color_range = [-5, 5]

    compare("marching_cubes_dynamic_attribute")

    iso.attribute = []

    compare("marching_cubes")


def test_marching_cubes_with_attribute_smoothed():
    global p, a

    prepare()

    iso = k3d.marching_cubes(p, attribute=a, level=0.0, flat_shading=False)

    pytest.plot += iso

    compare("marching_cubes_with_attribute_smoothed")


def test_marching_cubes_with_attribute_wireframe():
    global p, a

    prepare()

    iso = k3d.marching_cubes(p, attribute=a, level=0.0, wireframe=True)

    pytest.plot += iso

    compare("marching_cubes_with_attribute_wireframe")
