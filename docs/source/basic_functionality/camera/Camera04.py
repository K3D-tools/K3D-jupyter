import k3d
from k3d.platonic import Icosahedron
import numpy as np
from numpy import sin, cos, pi


def generate():
    plot = k3d.plot()
    plot += Icosahedron().mesh
    plot += Icosahedron((0, 2, 1), size=0.3).mesh

    plot.camera = [0.01, 0.01, 8] + \
                  [0, 0, 0] + \
                  [0, 1, 0]

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
