"""Khronos sample models read with k3d.glb / k3d.gltf, in every renderer.

Each of these models is built to show one failure plainly - a red cross, a wrong-way arrow - so
a reference that matches is also a render that passes the model's own test.
"""

import os
import warnings

import pytest

import k3d

from .plot_compare import compare, prepare

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gltf")


def load(name):
    path = os.path.join(ASSETS, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (k3d.glb if name.endswith(".glb") else k3d.gltf)(path)


@pytest.mark.parametrize("name", [
    "BoxTextured.glb",
    # vertex colours multiplied into a texture
    "VertexColorTest.glb",
    # wrapping per axis, double-sidedness
    "TextureSettingsTest.glb",
    # points, lines, loops, strips and fans
    "MeshPrimitiveModes.gltf",
    # mirrored nodes keep their faces outward
    "NegativeScaleTest.glb",
])
def test_gltf_sample(name):
    prepare()

    # the lines and points of MeshPrimitiveModes are white, as is the default background
    pytest.plot.background_color = 0x404850
    pytest.plot += load(name)

    compare("gltf_" + os.path.splitext(name)[0])
