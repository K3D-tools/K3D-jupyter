import numpy as np
import os

import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/plasma_wind.npz')

    data = np.load(filepath)
    L = float(data['L'])

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x05060D,
                    screenshot_scale=1.0,
                    axes_helper=0,
                    colorbar_object_id=0)

    plot += k3d.volume(data['density'].astype(np.float32),
                       samples=256, alpha_coef=40,
                       color_map=k3d.matplotlib_color_maps.jet,
                       color_range=[50, 550],
                       bounds=[-L, L, -L, L, -L, L])

    plot += k3d.points(data['stars'], shader='mesh',
                       point_sizes=data['star_size'], mesh_detail=4,
                       colors=np.array([0x2040FF, 0x3060FF, 0x30A0FF,
                                        0x2080E0, 0x5050FF], dtype=np.uint32))

    plot += k3d.lines(data['line_vertices'], data['line_indices'],
                      indices_type='segment', shader='mesh',
                      width=0.006, color=0xFFFFFF)

    plot.depth_peels = 4
    plot.renderer = 'advanced'
    plot.environment = 'moonless_golf'
    plot.camera = [3.1, -3.1, 2.0, 0, 0, 0, 0, 0, 1]

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
