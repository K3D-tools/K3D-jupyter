import numpy as np
import pytest

import k3d

from .plot_compare import compare, prepare


def test_stl():
    prepare()

    filename = "./test/assets/darth_vader.stl"

    with open(filename, "rb") as f:
        mesh = k3d.stl(
            f.read(),
            color=0x222222,
            flat_shading=True,
            transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        )

    pytest.plot += mesh

    compare("stl")


def test_stl_color():
    prepare()

    filename = "./test/assets/darth_vader.stl"

    with open(filename, "rb") as f:
        mesh = k3d.stl(
            f.read(),
            color=0xFF00FF,
            transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        )

    pytest.plot += mesh

    compare("stl_color")


def test_stl_wireframe():
    prepare()

    filename = "./test/assets/darth_vader.stl"

    with open(filename, "rb") as f:
        mesh = k3d.stl(
            f.read(),
            wireframe=True,
            transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        )

    pytest.plot += mesh

    compare("stl_wireframe")


def test_stl_smooth():
    prepare()

    filename = "./test/assets/darth_vader.stl"

    with open(filename, "rb") as f:
        mesh = k3d.stl(
            f.read(),
            flat_shading=False,
            color=0x222222,
            transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        )

    pytest.plot += mesh

    compare("stl_smooth")
