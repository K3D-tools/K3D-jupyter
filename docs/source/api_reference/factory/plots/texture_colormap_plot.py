import k3d
import numpy as np
from k3d.colormaps import matplotlib_color_maps


def generate():
    t = np.linspace(0, 1, 100)

    plt_texture = k3d.texture(color_map=matplotlib_color_maps.Jet,
                              attribute=t,
                              color_range=[0.15, 0.85])

    plot = k3d.plot()
    plot += plt_texture

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
