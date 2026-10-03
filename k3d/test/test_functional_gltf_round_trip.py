"""What K3D exports as glTF, K3D reads back the same.

The export writes the scene the browser built; k3d.glb reads it into objects again. Geometry has
to survive bit for bit, and colours to the 8-bit step: glTF stores them linear, K3D keeps the
displayed value, and both directions convert.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]], np.float32)
INDICES = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)


def _round_trip():
    pytest.headless.sync(hold_until_refreshed=True)
    blob = pytest.headless.get_gltf()

    # K3D's own exports are already z up
    return {obj.name: obj for obj in k3d.glb(blob, up="z")}


def test_geometry_colour_and_placement_come_back():
    prepare()

    translation = [0.5, -1.0, 2.0]
    pytest.plot += k3d.mesh(VERTICES, INDICES, color=0x336699, flat_shading=False,
                            name="plain", translation=translation)
    pytest.plot += k3d.mesh(VERTICES, INDICES, colors=[0xFF0000, 0x00FF00, 0x0000FF, 0x808080],
                            flat_shading=False, name="coloured")

    back = _round_trip()

    plain = back["plain"]
    np.testing.assert_array_equal(plain.vertices, VERTICES)
    np.testing.assert_array_equal(plain.indices, INDICES)
    np.testing.assert_allclose(np.asarray(plain.model_matrix)[:3, 3], translation, atol=1e-6)
    assert plain.color == 0x336699

    coloured = back["coloured"]
    np.testing.assert_array_equal(coloured.colors, [0xFF0000, 0x00FF00, 0x0000FF, 0x808080])
    assert coloured.color == 0xFFFFFF


def test_material_values_come_back():
    prepare()

    pytest.plot += k3d.mesh(VERTICES, INDICES, color=0xC08040, roughness=0.25, metalness=0.75,
                            opacity=0.5, emissive=0x204060, side="double", name="material")

    material = _round_trip()["material"]

    assert material.color == 0xC08040
    assert material.emissive == 0x204060
    assert material.roughness == pytest.approx(0.25)
    assert material.metalness == pytest.approx(0.75)
    assert material.opacity == pytest.approx(0.5)
    assert material.side == "double"
