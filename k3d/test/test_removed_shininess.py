"""The shininess tombstone: removed in 3.0.0, but loudly, because unknown constructor
kwargs are silently swallowed and a plain removal would be an invisible visual change."""

import numpy as np
import pytest
from traitlets import TraitError

import k3d

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def test_factory_kwarg_raises_with_conversion_hint():
    with pytest.raises(TraitError, match="roughness"):
        k3d.mesh(VERTICES, INDICES, shininess=50.0)


def test_assignment_raises():
    mesh = k3d.mesh(VERTICES, INDICES)

    with pytest.raises(TraitError, match="sqrt"):
        mesh.shininess = 500.0


def test_constructor_raises():
    from k3d.objects import Mesh

    with pytest.raises(TraitError, match="roughness"):
        Mesh(vertices=VERTICES, indices=INDICES, shininess=50.0)


def test_legacy_snapshot_shininess_converts():
    from k3d.objects.utils import create_object

    mesh = k3d.mesh(VERTICES, INDICES)
    state = mesh.get_binary()
    state.pop("roughness", None)
    state["shininess"] = 50.0

    restored = create_object(state)

    assert restored.roughness == pytest.approx(np.sqrt(2.0 / 52.0))


def test_replacement_traits_work():
    mesh = k3d.mesh(VERTICES, INDICES, roughness=0.4, metalness=0.7)

    assert mesh.roughness == 0.4
    assert mesh.metalness == 0.7

    mesh.roughness = 0.06
    assert mesh.roughness == 0.06
