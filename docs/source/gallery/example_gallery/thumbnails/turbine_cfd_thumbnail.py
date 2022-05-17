import os

import k3d
import numpy as np
import pyvista as pv
from k3d.colormaps import paraview_color_maps
from k3d.helpers import map_colors
from numpy.linalg import norm
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/cylinder.vtp')

    cylinder = pv.read(filepath)
    plt_vtk = k3d.vtk_poly_data(cylinder,
                                color=0x000000,
                                opacity=0.3)

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/cfd.npz')

    data = np.load(filepath)
    plt_vectors = k3d.vectors(data['o'], data['v'] * 3,
                              line_width=0.02,
                              colors=data['c'])

    plot = k3d.plot(screenshot_scale=1.0,
                    grid_visible=False,
                    axes_helper = 0)
    plot += plt_vtk
    plot += plt_vectors

    plot.camera = [3.0792, 14.6017, -8.8171,
                  -0.9959, 0.5287, -0.2337,
                  1, 0, 0]

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.85)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot