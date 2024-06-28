import numpy as np
import pytest

import k3d
from .plot_compare import prepare, compare

width = height = length = 10


def f(x, y, z):
    return (np.sin(float(x) / width * np.pi * 2.0) * 1.05,
            np.cos(float(y) / height * np.pi * 2.0) * 1.05,
            np.sin(float(z) / length * np.pi * 2.0) * 1.05)


colors = np.array(
    [[[(0xFF0000, 0x00FF00) for x in range(width)] for y in range(height)] for z in range(length)])
vectors = np.array(
    [[[f(x, y, z) for x in range(width)] for y in range(height)] for z in range(length)],
    dtype=np.float32)


def test_vector_field_3d():
    global vectors, colors, model_matrix

    prepare()

    vector_field = k3d.vector_field(vectors[::2, ::2, ::2], colors[::2, ::2, ::2], scale=1.5,
                                    line_width=0.001, head_size=1.5,
                                    transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += vector_field

    compare('vector_field_3d')


def test_vector_field_3d_no_head():
    global vectors, colors, model_matrix

    prepare()

    vector_field = k3d.vector_field(vectors, colors, scale=1.5, use_head=False, line_width=0.001,
                                    transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += vector_field

    compare('vector_field_3d_no_head')


def test_vector_field_3d_scale():
    global vectors, colors, model_matrix

    prepare()

    vector_field = k3d.vector_field(vectors, colors, scale=2.5, line_width=0.001,
                                    transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += vector_field

    compare('vector_field_3d_scale')
