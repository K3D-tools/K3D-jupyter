import numpy as np

import k3d
from k3d.colormaps import matplotlib_color_maps


def generate():
    t = np.linspace(-10, 10, 100, dtype=np.float32)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5

    vertices = np.vstack([x, y, z]).T

    line = k3d.line(vertices, width=0.2, shader='mesh',
                    color_map=matplotlib_color_maps.Jet,
                    attribute=t,
                    color_range=[-5, 5])

    plot = k3d.plot()
    plot += line

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
