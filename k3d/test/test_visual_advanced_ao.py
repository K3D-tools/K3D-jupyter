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


def points_scene(per_point_sizes=False):
    # a dense impostor cloud: sphere-sphere contacts are the only occlusion source,
    # so the image reacts to ao_radius/ao_strength only through the impostor depth pass
    prepare()
    pytest.plot.renderer = "advanced"

    rng = np.random.default_rng(3)
    n = 4000
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True) + 1e-9
    positions *= rng.uniform(0.2, 1.0, (n, 1)).astype(np.float32) ** (1.0 / 3.0)

    sizes = {}
    if per_point_sizes:
        sizes = {"point_sizes": (0.11 * rng.uniform(0.6, 1.4, n)).astype(np.float32)}

    pytest.plot += k3d.points(positions, point_size=0.11, shader="3d",
                              color=0xE3E9F3, roughness=0.35, **sizes)


def test_ao_points_3d():
    points_scene()
    pytest.plot.ao_radius = 0.05
    pytest.plot.ao_strength = 3.0

    compare("advanced_ao_points_3d", modes=("advanced",))


def test_ao_points_3d_responds_to_traits():
    # regression: with the (default) logarithmic depth buffer the impostor AO depth
    # pass used to store log-encoded depth, GTAO reconstructed a flat plane glued to
    # the near plane and the ao_* traits changed nothing in the image
    points_scene(per_point_sizes=True)
    pytest.plot.ao_radius = 0.01
    pytest.plot.ao_strength = 1.0
    pytest.headless.sync(hold_until_refreshed=True)

    weak = pytest.headless.get_screenshot(True)

    pytest.plot.ao_radius = 0.12
    pytest.plot.ao_strength = 5.0
    pytest.headless.sync(hold_until_refreshed=True)

    strong = pytest.headless.get_screenshot(True)

    assert weak != strong


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
