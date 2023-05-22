import numpy as np
import os

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'gallery/showcase/assets/streamlines_data.npz')

    data = np.load(filepath)

    plt_streamlines = k3d.line(data['lines'],
                               width=0.00007,
                               attribute=data['v'],
                               color_map=k3d.matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5],
                               shader='mesh')

    plt_mesh = k3d.mesh(data['vertices'], data['indices'],
                        opacity=0.25,
                        wireframe=True,
                        color=0x0002)

    plot = k3d.plot()
    plot += plt_streamlines
    plot += plt_mesh

    plot.camera = [0.0705, 0.0411, 0.0538,
                   0.0511, 0.0391, 0.0493,
                   -0.0798, 0.9872, 0.1265]

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
