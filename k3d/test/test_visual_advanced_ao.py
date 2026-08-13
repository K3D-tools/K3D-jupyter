"""Scenarios that only exist in the advanced renderer: ambient occlusion."""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare


def scene():
    prepare()
    pytest.plot.renderer = "advanced"

    # deep slits between slabs - a strong, structured occlusion pattern
    voxels = np.ones((16, 24, 24), dtype=np.uint8)
    voxels[4:, ::4, :] = 0

    pytest.plot += k3d.voxels(voxels, color_map=[0xC0C0C0])


def test_ao():
    scene()

    compare("advanced_ao", modes=("advanced",))


def test_menger_sponge():
    # the showcase scene (docs/source/gallery/showcase/plots/menger_sponge_plot.py):
    # a fractal of right-angle cavities - the strongest occlusion pattern we render
    prepare()

    def iterate(voxels, length, x, y, z):
        nl = length // 3

        if nl < 1:
            return

        margin = (nl - 1) // 2

        voxels[z - margin:z + margin + 1, y - margin:y + margin + 1, :] = 0
        voxels[z - margin:z + margin + 1, :, x - margin:x + margin + 1] = 0
        voxels[:, y - margin:y + margin + 1, x - margin:x + margin + 1] = 0

        for ix, iy, iz in np.ndindex((3, 3, 3)):
            if (1 if ix != 1 else 0) + (1 if iy != 1 else 0) + (1 if iz != 1 else 0) != 2:
                iterate(voxels, nl, x + (ix - 1) * nl, y + (iy - 1) * nl, z + (iz - 1) * nl)

    size = 3 ** 4
    voxels = np.ones((size, size, size))

    iterate(voxels, size, size // 2, size // 2, size // 2)

    pytest.plot += k3d.voxels(voxels.astype(np.uint8), color_map=(0xFDFE03), outlines=True)
    pytest.plot.grid_visible = False
    pytest.plot.camera = [98.5152, -60.0912, 88.9902,
                          43.4731, 37.6014, 31.5219,
                          -0.2226, 0.3405, 0.9135]
    pytest.headless.sync(hold_until_refreshed=True)

    compare("menger_sponge", camera_factor=None)


def test_ao_deterministic():
    scene()
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()

    first = pytest.headless.get_screenshot(True)
    second = pytest.headless.get_screenshot(True)

    assert first == second


def test_ao_no_strip_seams():
    scene()
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset()

    single = pytest.headless.get_screenshot(True)

    pytest.plot.rendering_steps = 6
    pytest.headless.sync(hold_until_refreshed=True)

    chunked = pytest.headless.get_screenshot(True)

    pytest.plot.rendering_steps = 1
    pytest.headless.sync(hold_until_refreshed=True)

    assert chunked == single
