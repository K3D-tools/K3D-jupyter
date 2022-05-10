import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps
from k3d.helpers import map_colors
from numpy.linalg import norm


def generate():
    p = np.linspace(-1, 1, 10)

    def f(x, y, z):
        return y * z, x * z, x * y

    vectors = np.array([[[f(x, y, z) for x in p] for y in p]
                       for z in p]).astype(np.float32)
    norms = np.apply_along_axis(norm, 1, vectors.reshape(-1, 3))

    plt_vector_field = k3d.vector_field(vectors,
                                        head_size=1.5,
                                        scale=2,
                                        bounds=[-1, 1, -1, 1, -1, 1])

    colors = map_colors(norms, matplotlib_color_maps.Turbo, [0, 1]).astype(np.uint32)
    plt_vector_field.colors = np.repeat(colors, 2)

    plot = k3d.plot()
    plot += plt_vector_field

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
