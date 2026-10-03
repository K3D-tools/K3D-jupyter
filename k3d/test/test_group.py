"""A Group holds objects and has no properties of its own; plot += takes one Drawable."""

import numpy as np
import pytest

import k3d
from k3d.objects import Group

TRIANGLE = ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])


def _pair():
    return k3d.mesh(*TRIANGLE, name="a"), k3d.points([[0, 0, 0]], name="b")


def test_a_group_has_no_properties_of_its_own():
    a, b = _pair()
    group = a + b

    for name, value in (("opacity", 0.5), ("visible", False), ("color", 3), ("transform", None)):
        with pytest.raises(AttributeError, match="no property"):
            setattr(group, name, value)


def test_a_group_still_gives_model_matrix_to_every_member():
    a, b = _pair()
    group = a + b
    matrix = np.diag([2, 2, 2, 1]).astype(np.float32)

    group.model_matrix = matrix

    np.testing.assert_array_equal(a.model_matrix, matrix)
    np.testing.assert_array_equal(b.model_matrix, matrix)


def test_a_group_is_indexed_and_counted():
    a, b = _pair()
    group = a + b

    assert len(group) == 2
    assert group[0] is a and group["b"] is b
    assert group["id"] == group.id
    with pytest.raises(KeyError):
        group["nobody"]


def test_a_group_holds_drawables_only():
    with pytest.raises(TypeError, match="Drawable"):
        Group([1, 2])


def test_plot_takes_one_drawable_and_says_how_to_combine():
    a, b = _pair()
    plot = k3d.plot()

    with pytest.raises(TypeError, match=r"a \+ b"):
        plot += [a, b]
    with pytest.raises(TypeError, match="Drawable"):
        plot += 3
    with pytest.raises(TypeError, match=r"-="):
        plot -= [a]
