import k3d
import numpy as np
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    t = np.linspace(-1.5, 1.5, 50, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing='ij')
    R = 1
    r = 0.5

    eq_heart = (x**2 + (9/4 * y**2) + z**2 - 1)**3 - \
        (x**2 * z**3) - (9/200 * y**2 * z**3)
    eq_torus = (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2 * (x**2 + y**2)

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)

    for i in range(3):
        level = 0 + i * 1.5

        plt_heart = k3d.marching_cubes(eq_heart, level=level,
                                       color=0xe31b23,
                                       xmin=-1.5, xmax=1.5,
                                       ymin=-1.5, ymax=1.5,
                                       zmin=-1.5, zmax=1.5,
                                       translation=[i * 3.5, 0, 0])
        plt_torus = k3d.marching_cubes(eq_torus, level=level,
                                       color=0x5aabac,
                                       xmin=-1.5, xmax=1.5,
                                       ymin=-1.5, ymax=1.5,
                                       zmin=-1.5, zmax=1.5,
                                       translation=[i * 3.5, 0, -3.5])

        plot += plt_heart
        plot += plt_torus

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
