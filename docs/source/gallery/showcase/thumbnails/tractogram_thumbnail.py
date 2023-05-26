import numpy as np
import os

import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/tractogram.npz')

    data = np.load(filepath)['data']

    v = data.copy()
    v[1:] = (v[1:] - v[:-1])
    v = np.absolute((v / np.linalg.norm(v, axis=1)[..., np.newaxis]))
    v = (v * 255).astype(np.int32)
    colors = np.sum((v * np.array([1, 256, 256 * 256])), axis=1).astype(np.uint32)

    streamlines = k3d.line(data, shader='thick', colors=colors, width=0.005)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0,
                    screenshot_scale=1.0,
                    axes_helper=0)

    plot += streamlines

    plot.camera = [-60.0, 135.0, 45.0,
                   -1.0, 0.5, -5.0,
                   0.0, -0.25, 1.0]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
