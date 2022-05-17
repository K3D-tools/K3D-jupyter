import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


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

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_points

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.85)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
