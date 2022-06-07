import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    t = np.linspace(-10, 10, 100, dtype=np.float32)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5

    vertices = np.vstack([x, y, z]).T

    line = k3d.line(vertices, width=0.2, shader='mesh',
                    color_map=matplotlib_color_maps.Jet,
                    attribute=t,
                    color_range=[-5, 5])

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += line

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
