import k3d
import numpy as np


def generate():
    xs = []
    for i in range(5):
        xs.append(np.random.randn(150, 3).astype(np.float32))

    plt_points_red = k3d.points(xs[0],
                                point_size=0.2,
                                color=0xff0000,
                                name='Red data')
    plt_points_green = k3d.points(xs[1] - 0.9,
                                  point_size=0.2,
                                  color=0x00ff00,
                                  name='Green data')
    plt_points_blue = k3d.points(xs[2] + 0.5,
                                point_size=0.2,
                                color=0x0000ff,
                                name='Blue data')
    plt_points_yellow = k3d.points(xs[3] - 1.5,
                                  point_size=0.2,
                                  color=0xffff00,
                                  name='Yellow data')
    plt_points_black = k3d.points(xs[4] + 1,
                                  point_size=0.2,
                                  color=0x000000,
                                  name='Black data')

    plot = k3d.plot(name='Multidata point cloud')
    plot += plt_points_red
    plot += plt_points_green
    plot += plt_points_blue
    plot += plt_points_yellow
    plot += plt_points_black

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
