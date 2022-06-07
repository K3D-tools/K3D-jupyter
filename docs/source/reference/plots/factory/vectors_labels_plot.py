import k3d
import numpy as np


def generate():
    o = np.array([[1, 2, 3],
                  [2, -3, 0]]).astype(np.float32)

    v = np.array([[1, 1, 1],
                  [-4, 2, 3]]).astype(np.float32)

    labels = ['(1, 1, 1)', '(2, -3, 0)']

    plt_vectors = k3d.vectors(origins=o,
                              vectors=v,
                              origin_color=0x000000,
                              head_color=0x488889,
                              line_width=0.2,
                              use_head=False,
                              labels=labels)

    plot = k3d.plot()
    plot += plt_vectors

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
