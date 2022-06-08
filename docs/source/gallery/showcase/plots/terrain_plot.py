import numpy as np
from pyvista import examples

import k3d


def generate():
    dem = examples.download_crater_topo()
    data = dem.get_array(0).reshape(dem.dimensions[::-1])[0, :, :].astype(np.float32)

    plot = k3d.plot()

    obj = k3d.surface(data,
                      attribute=data,
                      color_map=k3d.colormaps.matplotlib_color_maps.viridis,
                      flat_shading=False,
                      xmin=dem.bounds[0],
                      xmax=dem.bounds[1],
                      ymin=dem.bounds[2],
                      ymax=dem.bounds[3])

    plot += obj

    plot.camera = [1823987, 5637535, 7254,
                   1821623, 5647785, -633,
                   -0.12, 0.54, 0.83]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
