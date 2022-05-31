import k3d
import pytest
from .plot_compare import *
import numpy as np
from math import sqrt, sin, cos

color_map = (0xffff00, 0xff0000, 0x00ff00)


def r(x, y, z, width=100, height=100, length=100):
    r = sqrt((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2) + (
            z - length / 2) * (z - length / 2))
    r += sin(x / 2) * 3
    r += cos(y / 10) * 5

    return r


def f(x, y, z, width=100, height=100, length=100):
    return 0 if r(x, y, z, width, height, length) > width / 2 else (
        1 if z + sin(x / 20) * 10 > length / 2 else 2)


voxels = np.array(
    [[[f(x, y, z, 100, 100, 100) for x in range(100)] for y in range(100)] for z in range(100)])


def test_voxels():
    global color_map, voxels

    prepare()

    obj = k3d.voxels(voxels.astype(np.uint8), color_map, outlines=False)

    pytest.plot += obj

    compare('voxels')


def test_voxels_sparse():
    global color_map, voxels

    prepare()

    sparse_data = []

    for val in np.unique(voxels):
        if val != 0:
            z, y, x = np.where(voxels == val)
            sparse_data.append(
                np.dstack((x, y, z, np.full(x.shape, val))).reshape(-1, 4).astype(np.uint16))

    sparse_data = np.vstack(sparse_data)
    obj = k3d.sparse_voxels(sparse_data, (100, 100, 100), color_map, outlines=False)

    pytest.plot += obj

    compare('voxels_sparse')


def test_voxels_wireframe():
    global color_map, voxels

    prepare()

    obj = k3d.voxels(voxels.astype(np.uint8), color_map, outlines=False, wireframe=True)

    pytest.plot += obj

    compare('voxels_wireframe')


def test_voxels_outline():
    global color_map, voxels

    prepare()

    obj = k3d.voxels(voxels.astype(np.uint8), color_map, outlines=True)

    pytest.plot += obj

    compare('voxels_outline')

    obj.outlines_color = 0xff0000

    compare('voxels_outline_dynamic_outlines_color')


def test_voxels_outline_opacity():
    global color_map, voxels

    prepare()

    obj = k3d.voxels(voxels.astype(np.uint8), color_map, outlines=True, opacity=0.5)

    pytest.plot += obj

    compare('voxels_outline_opacity')


def test_voxels_outline_clipping_plane():
    global color_map, voxels

    prepare()

    obj = k3d.voxels(voxels.astype(np.uint8), color_map, outlines=True)

    pytest.plot += obj
    pytest.plot.clipping_planes = [
        [-1, 0, -1, 50]
    ]

    compare('voxels_outline_clipping_plane')


def test_voxels_box():
    global color_map, voxels

    prepare()

    width, height, length = 200, 50, 100
    box_voxels = voxels = np.array(
        [[[f(x, y, z, width, height, length) for x in range(width)] for y in range(height)] for z in
         range(length)])

    obj = k3d.voxels(box_voxels.astype(np.uint8), color_map, outlines=True)

    pytest.plot += obj

    compare('voxels_box')
