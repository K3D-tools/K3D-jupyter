import k3d
import numpy as np
from numpy import sin
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    t = np.linspace(-5, 5, 100, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing='ij')

    scalars = sin(x*y + x*z + y*z) + sin(x*y) + sin(y*z) + sin(x*z) - 1

    marching = k3d.marching_cubes(scalars, level=0.0,
                                  color=0x0e2763,
                                  opacity=0.25,
                                  xmin=0, xmax=1,
                                  ymin=0, ymax=1,
                                  zmin=0, zmax=1,
                                  compression_level=9,
                                  flat_shading=False)

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += marching

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
