import numpy as np

import k3d


def generate():
    t = np.linspace(-10, 10, 100, dtype=np.float32)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5

    vertices = np.vstack([x, y, z]).T

    line = k3d.line(vertices, width=0.1, color=0xff99cc)

    plot = k3d.plot()
    plot += line

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
