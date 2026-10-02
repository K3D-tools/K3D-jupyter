"""Bounding boxes on the Python side feed camera_auto_fit and get_auto_grid, so a wrong one is a
wrong first view, and one that raises takes get_auto_camera down with it.

None of this touches a browser: the boxes are computed from the traits alone.
"""

import numpy as np
import pytest

import k3d


def test_vectors_box_spans_origin_to_tip():
    """The components of a vector are not positions; the tip is origin plus vector."""
    v = k3d.vectors([[10, 10, 10]], [[1, 0, 0]])

    assert np.allclose(v.get_bounding_box(), [10, 11, 10, 10, 10, 10])


@pytest.mark.parametrize("vectors, expected", [
    (
        np.array([[[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                  [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [8, -4, 2]]]], dtype=np.float32),
        [-0.5, 4, -2, 0, -0.5, 1],
    ),
    (
        np.array([[[0, 0], [0, 0]], [[0, 0], [-8, 6]]], dtype=np.float32),
        [-4, 0, -0.5, 3, 0, 0],
    ),
])
def test_vector_field_box_spans_grid_origins_and_arrow_tips(vectors, expected):
    field = k3d.vector_field(vectors, scale=2)
    plot = k3d.plot()
    plot += field

    assert np.allclose(field.get_bounding_box(), expected)
    assert np.allclose(plot.get_auto_grid(), expected)
    expected_center = (np.array(expected[::2]) + np.array(expected[1::2])) / 2
    assert np.allclose(plot.get_auto_camera()[3:6], expected_center)


def test_voxels_honours_separate_bound_arguments():
    """voxels() used to always inject its own "bounds" into kwargs, even when the caller
    passed xmin/xmax/... instead of bounds. process_transform_arguments only falls back to
    those separate arguments when "bounds" is absent from kwargs, so they were silently
    dropped and the object was placed at the data-derived default box instead.
    """
    data = np.ones((4, 4, 4), np.uint8)

    v = k3d.voxels(data, xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1)

    assert np.allclose(v.get_bounding_box(), [-1, 1, -1, 1, -1, 1])


def test_voxels_without_bounds_keeps_data_derived_box():
    """With no bounds and no separate arguments, the box still comes from the data shape."""
    data = np.ones((4, 5, 6), np.uint8)

    v = k3d.voxels(data)

    assert np.allclose(v.get_bounding_box(), [0, 6, 0, 5, 0, 4])


def test_auto_grid_is_min_max_interleaved():
    p = k3d.plot()
    p += k3d.points(np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32))

    assert np.allclose(p.get_auto_grid(), [0, 2, 0, 4, 0, 6])


def test_auto_grid_skips_2d_text_and_spans_every_frame():
    p = k3d.plot()
    p += k3d.points(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
    p += k3d.text2d("title")
    p += k3d.text("far", position=[5, 0, 0])
    p += k3d.points({
        "0.0": np.array([[0, 0, 0]], dtype=np.float32),
        "1.0": np.array([[0, 0, 9]], dtype=np.float32),
    })

    grid = p.get_auto_grid()

    assert grid.shape == (6,)
    assert np.allclose(grid, [0, 5, 0, 1, 0, 9])


def test_auto_camera_survives_a_2d_overlay():
    p = k3d.plot()
    p += k3d.text2d("only an overlay")
    p += k3d.points(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))

    camera = p.get_auto_camera()

    assert len(camera) == 9
    assert np.all(np.isfinite(camera))
