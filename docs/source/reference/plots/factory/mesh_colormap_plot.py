import numpy as np
from matplotlib.tri import Triangulation

import k3d
from k3d.colormaps import matplotlib_color_maps


def generate():
    n_radii = 8
    n_angles = 36

    radii = np.linspace(0.125, 1.0, n_radii, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_angles,
                         endpoint=False, dtype=np.float32)[..., np.newaxis]

    x = np.append(np.float32(0), (radii * np.cos(angles)).flatten())
    y = np.append(np.float32(0), (radii * np.sin(angles)).flatten())
    z = np.sin(-x * y)

    vertices = np.vstack([x, y, z]).T
    indices = Triangulation(x, y).triangles.astype(np.uint32)

    plt_mesh = k3d.mesh(vertices, indices,
                        color_map=matplotlib_color_maps.Jet,
                        attribute=z,
                        color_range=[-1.1, 2.01])

    plot = k3d.plot()
    plot += plt_mesh

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
