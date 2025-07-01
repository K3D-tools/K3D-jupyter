import numpy as np

import k3d


def generate():
    o = np.array([[0, 0, 0],
                  [2, 3, 4]]).astype(np.float32)

    v = np.array([[1, 1, 1],
                  [-2, -2, -2]]).astype(np.float32)

    plt_vectors = k3d.vectors(origins=o,
                              vectors=v,
                              colors=[0x000000, 0xde49a1,
                                      0x000000, 0x40826d])

    plot = k3d.plot()
    plot += plt_vectors

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
