"""Bounding boxes on the Python side feed camera_auto_fit and get_auto_grid, so a wrong one is a
wrong first view, and one that raises takes get_auto_camera down with it.

None of this touches a browser: the boxes are computed from the traits alone.
"""

import numpy as np

import k3d


def test_vectors_box_spans_origin_to_tip():
    """The components of a vector are not positions; the tip is origin plus vector."""
    v = k3d.vectors([[10, 10, 10]], [[1, 0, 0]])

    assert np.allclose(v.get_bounding_box(), [10, 11, 10, 10, 10, 10])


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
