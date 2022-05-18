import os

import k3d
import numpy as np
from k3d.headless import k3d_remote, get_headless_driver


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
                    background_color=0x87ceeb,
                    screenshot_scale=1.0,
                    axes_helper = 0)
    plot += plt_points

    plot.camera = [20.84, -3.06, 6.96,
                   0.67, 0.84, 3.79,
                   0.0, 0.0, 1.0]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    # headless.camera_reset(0.85)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot

