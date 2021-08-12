import numpy as np
import k3d


def generate():
    plot = k3d.plot()

    Nx, Ny = 50, 60
    xmin, xmax, ymin, ymax = -3, 3, 0, 3
    x = np.linspace(xmin, xmax, Nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, Ny, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    f = np.sin(x ** 2 + y ** 2)
    plt_surface = k3d.surface(f, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plot += plt_surface

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
