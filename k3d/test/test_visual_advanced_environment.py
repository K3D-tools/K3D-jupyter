"""Scenarios that only exist in the advanced renderer: environment presets and backgrounds."""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

VERTICES = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
INDICES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)


def scene():
    prepare()
    pytest.plot.renderer = "advanced"
    pytest.plot += k3d.mesh(VERTICES, INDICES, color=0x2244AA, flat_shading=False, roughness=0.1)


def test_environment_presets():
    scene()

    for preset in ("neutral", "studio", "outdoor"):
        pytest.plot.environment = preset
        compare("advanced_environment_" + preset, modes=("advanced",))


def test_environment_background():
    scene()
    pytest.plot.environment = "studio"
    pytest.plot.show_environment = True

    compare("advanced_environment_background", modes=("advanced",))


def test_environment_rotation():
    scene()
    pytest.plot.environment = "outdoor"

    compare("advanced_environment_rotation_0", modes=("advanced",))

    pytest.plot.environment_rotation = np.pi

    compare("advanced_environment_rotation_pi", modes=("advanced",))


def test_environment_custom_map():
    scene()

    # a hard split: red radiance on one side, blue on the other
    env = np.zeros((32, 64, 3), dtype=np.float32)
    env[:, :32, 0] = 2.0
    env[:, 32:, 2] = 2.0
    pytest.plot.environment = env

    compare("advanced_environment_custom", modes=("advanced",))


def test_environment_catalog():
    scene()
    pytest.plot.environment = "brown_photostudio_02"

    compare("advanced_environment_catalog", modes=("advanced",))
