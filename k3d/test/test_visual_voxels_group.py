import numpy as np
import pytest
import skimage.measure

import k3d
from .plot_compare import *


def generate(dim):
    data = np.zeros((dim, dim, dim), dtype=np.uint8)

    x = np.linspace(-0.5, 0.5, dim, dtype=np.float32)
    y = np.linspace(-0.5, 0.5, dim, dtype=np.float32)
    z = np.linspace(-0.5, 0.5, dim, dtype=np.float32)

    x, y, z = np.meshgrid(x, y, z)

    c, s = np.cos(1.5 * x), np.sin(1.5 * x)

    my = y * c - z * s
    mz = y * s + z * c

    my = np.fmod(my + 0.5, 0.333) * 3 - 0.5
    mz = np.fmod(mz + 0.5, 0.333) * 3 - 0.5

    displace = np.sin(60.0 * x) * np.sin(60.0 * my) * np.sin(60.0 * mz) * 0.1

    data = np.sqrt(my ** 2 + mz ** 2) * (2.5 + 0.8 * np.sin(x * 50)) + displace

    return (data < 0.25).astype(np.uint8)


dim = 256
data = generate(dim)

chunk_size = 16
voxelsGroup = []

for z, y, x in zip(*np.where(
        skimage.measure.block_reduce(data, (chunk_size, chunk_size, chunk_size), np.max) > 0)):
    chunk = {
        "voxels": data[z * chunk_size:(z + 1) * chunk_size,
                  y * chunk_size:(y + 1) * chunk_size,
                  x * chunk_size:(x + 1) * chunk_size],
        "coord": np.array([x, y, z]) * chunk_size,
        "multiple": 1
    }

    voxelsGroup.append(chunk)


def test_voxels_group():
    global voxelsGroup, data

    prepare()

    space_size = np.array(data.shape, dtype=np.uint32)[::-1]
    obj = k3d.voxels_group(space_size, voxelsGroup)

    pytest.plot += obj

    compare('voxels_group')

    obj.opacity = 0.2

    compare('voxels_group_dynamic_opacity')


def test_voxels_group_wireframe():
    global voxelsGroup, data

    prepare()

    space_size = np.array(data.shape, dtype=np.uint32)[::-1]
    obj = k3d.voxels_group(space_size, voxelsGroup, wireframe=True, opacity=0.25)

    pytest.plot += obj

    compare('voxels_group_wireframe', camera_factor=0.5)


def test_voxels_group_opacity():
    global voxelsGroup, data

    prepare()

    space_size = np.array(data.shape, dtype=np.uint32)[::-1]
    obj = k3d.voxels_group(space_size, voxelsGroup, opacity=0.5)

    pytest.plot += obj

    compare('voxels_group_opacity')
