import k3d
import numpy as np


def generate():
    sparse_voxels = np.array([[1, 0, 0, 1],
                              [0, 1, 0, 1],
                              [0, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 0, 2],
                              [0, 0, 1, 2],
                              [1, 0, 1, 2]]).astype(np.uint16)

    plt_sparse_voxels = k3d.sparse_voxels(sparse_voxels,
                                          space_size=[2, 2, 2],
                                          color_map=[0xfdc192, 0xa15525],
                                          outlines_color=0xffffff)

    plot = k3d.plot()
    plot += plt_sparse_voxels

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
