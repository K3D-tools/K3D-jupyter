import numpy as np
import k3d


def generate():
    np.random.seed(0)

    x = np.random.randn(100,3).astype(np.float32)
    plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=x, point_size=0.2, shader='3d')
    plot += plt_points

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
