import numpy as np
import k3d
from numpy import sin, cos, pi

def generate():
    plot = k3d.plot()

    T = 1.6
    r = 4.77
    zmin, zmax = -r, r
    xmin, xmax = -r, r
    ymin, ymax = -r, r
    Nx, Ny, Nz = 37, 37, 37

    x = np.linspace(xmin, xmax, Nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, Ny, dtype=np.float32)
    z = np.linspace(zmin, zmax, Nz, dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    p = 2 - (cos(x + T * y) + cos(x - T * y) + cos(y + T * z) +
             cos(y - T * z) + cos(z - T * x) + cos(z + T * x))
    plt_iso = k3d.marching_cubes(p, compression_level=9, xmin=xmin, xmax=xmax,
                                 ymin=ymin, ymax=ymax,
                                 zmin=zmin, zmax=zmax, level=0.0,
                                 flat_shading=False)
    plot += plt_iso

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
