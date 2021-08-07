import k3d
import numpy as np


def generate():
    np.random.seed(0)
    x = np.random.randn(1000, 3).astype(np.float32)
    plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=x, point_size=0.2)
    plot += plt_points
    plt_points.shader = '3d'

    f = (np.sum(x ** 3 - .1 * x ** 2, axis=1))
    colormap = k3d.colormaps.basic_color_maps.WarmCool
    colors = k3d.helpers.map_colors(f, colormap, [-2, .1])
    colors = colors.astype(np.uint32)
    plt_points.colors = colors

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
