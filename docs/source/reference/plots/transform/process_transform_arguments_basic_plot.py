import numpy as np

import k3d
from k3d.transform import process_transform_arguments


def generate():
    vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 3, 1], [3, 2, 1]]

    plt_tetra = k3d.mesh(vertices, indices,
                         color=0x00a86b)
    plt_tetra1 = k3d.mesh(vertices, indices,
                          color=0x4ca5ff)

    process_transform_arguments(plt_tetra1,
                                translation=[1, 3, 0],
                                rotation=[1, np.pi / 2, 0, 0],
                                scaling=[0.5, 1, 1.5])

    plot = k3d.plot()
    plot += plt_tetra
    plot += plt_tetra1

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
