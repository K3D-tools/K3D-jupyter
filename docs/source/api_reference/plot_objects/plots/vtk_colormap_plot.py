import os

import k3d
import pyvista as pv
from k3d.colormaps import matplotlib_color_maps


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/bunny.vtp')

    data = pv.read(filepath)

    plt_vtk = k3d.vtk_poly_data(data,
                                color_attribute=('Normals', 0, 1),
                                color_map=matplotlib_color_maps.Rainbow,
                                model_matrix=(1.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0,
                                              0.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += plt_vtk

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
