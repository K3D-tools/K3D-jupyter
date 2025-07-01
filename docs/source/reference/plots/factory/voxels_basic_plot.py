import numpy as np

import k3d


def generate():
    voxels = np.array([[[0, 1],
                        [1, 2]],
                       [[2, 2],
                        [1, 1]]]).astype(np.uint8)

    plt_voxels = k3d.voxels(voxels,
                            color_map=[0xfdc192, 0xa15525],
                            outlines_color=0xffffff)

    plot = k3d.plot()
    plot += plt_voxels

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
