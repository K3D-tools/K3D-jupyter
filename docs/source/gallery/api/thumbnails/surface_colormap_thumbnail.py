import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    x = np.linspace(-5, 5, 100, dtype=np.float32)
    y = np.linspace(-5, 5, 100, dtype=np.float32)

    x, y = np.meshgrid(x, y)
    f = ((x**2 - 1) * (y**2 - 4) + x**2 + y**2 - 5) / (x**2 + y**2 + 1)**2

    plt_surface = k3d.surface(f * 2,
                              xmin=-5, xmax=5,
                              ymin=-5, ymax=5,
                              compression_level=9,
                              color_map=matplotlib_color_maps.Coolwarm_r,
                              attribute=f, color_range=[-1, 0.5])

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_surface

    plot.camera = [-10, 9, -2.5,
                   0, 0, 0,
                   0, 0, -1]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.8)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot

