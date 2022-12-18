import numpy as np
import os
import pyvista as pv

import k3d


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

    plot = k3d.plot()
    plot += plt_vtk
    plot += plt_vectors

    plot.camera = [3.0792, 14.6017, -8.8171,
                   -0.9959, 0.5287, -0.2337,
                   1, 0, 0]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
