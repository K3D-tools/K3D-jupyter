import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps


def generate():
    x = np.random.randn(10000, 3).astype(np.float32)
    f = (np.sum(x ** 3 - .1 * x ** 2, axis=1))

    plt_points = k3d.points(positions=x,
                            point_size=0.1,
                            shader='flat',
                            opacity=0.7,
                            color_map=matplotlib_color_maps.Coolwarm,
                            attribute=f,
                            color_range=[-2, 1])

    plot = k3d.plot()
    plot += plt_points

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
