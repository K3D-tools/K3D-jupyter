import numpy as np

import k3d
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    t = np.linspace(0, 1, 100)

    plt_texture = k3d.texture(color_map=matplotlib_color_maps.Jet,
                              attribute=t,
                              color_range=[0.15, 0.85])

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_texture

    plot.camera = [0, 0, 1.5,
                   0, 0, 0,
                   0, 1, 0]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
