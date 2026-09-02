import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from renderers_scene import material_grid_plot


def generate():
    plot = material_grid_plot()

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
