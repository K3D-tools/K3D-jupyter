import numpy as np
import pytest

import k3d
from .plot_compare import prepare, compare

T = 1.618033988749895
r = 4.77
zmin, zmax = -r, r
xmin, xmax = -r, r
ymin, ymax = -r, r
Nx, Ny, Nz = 77, 77, 77

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
z = np.linspace(zmin, zmax, Nz)
x, y, z = np.meshgrid(x, y, z, indexing='ij')
p = 2 - (np.cos(x + T * y) + np.cos(x - T * y) + np.cos(y + T * z) + np.cos(y - T * z) + np.cos(
    z - T * x) + np.cos(z + T * x)).astype(np.float32)


def test_marching_cubes():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0)

    pytest.plot += iso

    compare('marching_cubes')

    iso.level = 0.1

    compare('marching_cubes_dynamic_level')


def test_marching_cubes_smoothed():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0, flat_shading=False)

    pytest.plot += iso

    compare('marching_cubes_smoothed')


def test_marching_cubes_wireframe():
    global p
    prepare()

    iso = k3d.marching_cubes(p, level=0.0, wireframe=True)

    pytest.plot += iso

    compare('marching_cubes_wireframe')


def test_marching_cubes_non_uniformly_spaced():
    prepare()

    ax = np.logspace(0, np.log10(xmax - xmin + 1), Nx, endpoint=True) + xmin - 1
    ay = np.logspace(0, np.log10(ymax - ymin + 1), Nx, endpoint=True) + ymin - 1
    az = np.logspace(0, np.log10(zmax - zmin + 1), Nx, endpoint=True) + zmin - 1

    x, y, z = np.meshgrid(ax, ay, az, indexing='ij')
    p = 2 - (np.cos(x + T * y) + np.cos(x - T * y) + np.cos(y + T * z) + np.cos(y - T * z) + np.cos(
        z - T * x) + np.cos(z + T * x)).astype(np.float32)

    iso = k3d.marching_cubes(p, level=0.0, color=1,
                             spacings_x=(ax[1:] - ax[:-1]).astype(np.float32),
                             spacings_y=(ay[1:] - ay[:-1]).astype(np.float32),
                             spacings_z=(az[1:] - az[:-1]).astype(np.float32),
                             bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
                             opacity=0.15,
                             wireframe=True)
    pytest.plot += iso

    compare('marching_cubes_non_uniformly_spaced')
