import numpy as np
import os

import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/streamlines_data.npz')

    data = np.load(filepath)

    v = data['v']
    lines = data['lines']
    vertices = data['vertices']
    indices = data['indices']

    plt_streamlines = k3d.line(lines,
                               width=0.00007,
                               attribute=v,
                               color_map=k3d.matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5],
                               shader='mesh')

    plt_mesh = k3d.mesh(vertices, indices,
                        opacity=0.25,
                        wireframe=True,
                        color=0x0002)

    plot = k3d.plot(grid_visible=False,
                    screenshot_scale=1.0,
                    axes_helper=0)
    plot += plt_streamlines
    plot += plt_mesh

    plot.camera = [0.0705, 0.0411, 0.0538,
                   0.0511, 0.0391, 0.0493,
                   -0.0798, 0.9872, 0.1265]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.85)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
