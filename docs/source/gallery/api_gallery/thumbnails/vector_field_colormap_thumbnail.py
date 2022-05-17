import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps
from k3d.helpers import map_colors
from numpy.linalg import norm
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    p = np.linspace(-1, 1, 10)

    def f(x, y, z):
        return y * z, x * z, x * y

    vectors = np.array([[[f(x, y, z) for x in p] for y in p]
                       for z in p]).astype(np.float32)
    norms = np.apply_along_axis(norm, 1, vectors.reshape(-1, 3))

    plt_vector_field = k3d.vector_field(vectors,
                                        head_size=1.5,
                                        scale=2,
                                        bounds=[-1, 1, -1, 1, -1, 1])

    colors = map_colors(norms, matplotlib_color_maps.Turbo, [0, 1]).astype(np.uint32)
    plt_vector_field.colors = np.repeat(colors, 2)

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_vector_field

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
