import os
import pyvista as pv

import k3d
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/cow.vtp')

    data = pv.read(filepath)

    plt_vtk = k3d.vtk_poly_data(data)

    lines = k3d.lines(plt_vtk.vertices, plt_vtk.indices,
                      shader='mesh', width=0.025,
                      color=0xc6884b,
                      model_matrix=(1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += lines

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
