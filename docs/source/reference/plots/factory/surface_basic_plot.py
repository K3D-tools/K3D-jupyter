import numpy as np

import k3d

sin, sqrt = np.sin, np.sqrt


def generate():
    x = np.linspace(-5, 5, 100, dtype=np.float32)
    y = np.linspace(-5, 5, 100, dtype=np.float32)

    x, y = np.meshgrid(x, y)
    f = sin(sqrt(x ** 2 + y ** 2))

    plt_surface = k3d.surface(f,
                              color=0x006394,
                              wireframe=True,
                              xmin=0, xmax=10,
                              ymin=0, ymax=10)

    plot = k3d.plot()
    plot += plt_surface

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
