import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps


def generate():
    x = np.linspace(-5, 5, 100, dtype=np.float32)
    y = np.linspace(-5, 5, 100, dtype=np.float32)

    x, y = np.meshgrid(x, y)
    f = ((x**2 - 1) * (y**2 - 4) + x**2 + y**2 - 5) / (x**2 + y**2 + 1)**2

    plt_surface = k3d.surface(f * 2,
                              xmin=-5, xmax=5,
                              ymin=-5, ymax=5,
                              compression_level=9,
                              color_map=matplotlib_color_maps.Coolwarm_r,
                              attribute=f, color_range=[-1, 0.5])

    plot = k3d.plot()
    plot += plt_surface

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
