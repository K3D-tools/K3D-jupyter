import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from renderers_scene import material_grid_plot


def generate():
    plot = material_grid_plot(renderer='cinematic')
    plot.environment = 'studio'
    # this embed accumulates in the reader's browser, and is denoised there
    plot.cinematic_samples = 64
    plot.cinematic_denoise = 0.8
    plot.cinematic_bounces = 4

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
