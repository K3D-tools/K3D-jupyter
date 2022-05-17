import os

import k3d
import pyvista as pv
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../api_reference/plot_objects/assets/bunny.vtp')

    data = pv.read(filepath)

    plt_vtk = k3d.vtk_poly_data(data,
                                color_attribute=('Normals', 0, 1),
                                color_map=matplotlib_color_maps.Rainbow,
                                model_matrix=(1.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0,
                                              0.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_vtk

    headless = k3d_remote(plot, get_headless_driver(), width=600, height=370)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
