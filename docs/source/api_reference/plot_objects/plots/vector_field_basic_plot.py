import k3d
import numpy as np


def generate():
    def f(x, y):
        return np.sin(y), np.sin(x)

    H = W = 10

    vectors = np.array([[f(x, y) for x in range(W)]
                       for y in range(H)]).astype(np.float32)

    plt_vector_field = k3d.vector_field(vectors,
                                        color=0xed6a5a)

    plot = k3d.plot()
    plot += plt_vector_field

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
