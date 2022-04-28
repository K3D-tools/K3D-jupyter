import os

import k3d
import numpy as np


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/point_cloud.npz')

    data = np.load(filepath)['arr_0']

    plt_points = k3d.points(data[:, 0:3],
                            data[:, 4].astype(np.uint32),
                            point_size=0.15,
                            shader="flat")

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x87ceeb)
    plot += plt_points

    plot.camera = [20.84, -3.06, 6.96,
                   0.67, 0.84, 3.79,
                   0.0, 0.0, 1.0]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
