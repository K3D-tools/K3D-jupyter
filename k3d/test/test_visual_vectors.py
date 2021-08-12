import k3d
import pytest
from .plot_compare import *
import numpy as np


def test_vectors():
    prepare()

    origins = [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vectors = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    colors = [0xff0000, 0x0000ff, 0x0000ff, 0xff0000, 0x0000ff, 0xff0000]

    vectors = k3d.vectors(origins, vectors, colors=colors, labels=[], label_size=1.5)
    pytest.plot += vectors

    compare('vectors')


def test_vectors_labels():
    prepare()

    origins = [1.2, 1.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vectors = [1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 1.5, 1.5, 1.5]
    colors = [0xff0000, 0x0000ff, 0x0000ff, 0xff0000, 0x0000ff, 0xff0000]

    vectors = k3d.vectors(origins, vectors, colors=colors, label_size=1.5,
                          labels=['aa', 'bb', 'cc'])
    pytest.plot += vectors

    compare('vectors_labels', False)

    vectors.head_size = 3.0

    compare('vectors_labels_dynamic_head_size', False)


def test_vectors_advance():
    prepare()

    n = 20
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    z = np.linspace(-5, 5, n)
    xx, yy, zz = np.meshgrid(x, y, z)
    uu, vv, ww = zz, yy, xx
    xx, yy, zz, uu, vv, ww = [t.flatten().astype(np.float32) for t in [xx, yy, zz, uu, vv, ww]]
    scale = 0.25
    magnitude = np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)
    vectors = np.array((uu, vv, ww)).T * scale
    origins = np.array((xx, yy, zz)).T
    colors = k3d.helpers.map_colors(magnitude, k3d.matplotlib_color_maps.Plasma, [])
    vec_colors = np.zeros(2 * len(colors))
    for i, c in enumerate(colors):
        vec_colors[2 * i] = c
        vec_colors[2 * i + 1] = c
    vec_colors = vec_colors.astype(np.uint32)
    vector_field = k3d.vectors(
        origins=origins - vectors / 2,
        vectors=vectors,
        colors=vec_colors,
    )

    pytest.plot += vector_field

    compare('vectors_advance')
