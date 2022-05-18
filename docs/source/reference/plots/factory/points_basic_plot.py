import k3d
import numpy as np


def generate():
    x = np.random.randn(1000, 3).astype(np.float32)

    plt_points = k3d.points(positions=x,
                            point_size=0.2,
                            shader='3d',
                            color=0x3f6bc5)

    plot = k3d.plot()
    plot += plt_points

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
