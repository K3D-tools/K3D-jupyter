import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from renderers_scene import material_grid_plot  # noqa: E402


def generate():
    plot = material_grid_plot(renderer='cinematic')
    plot.environment = 'studio'
    # a page embed accumulates in the reader's browser: enough samples to read
    # the light, few enough to settle in a moment
    plot.cinematic_samples = 24
    plot.cinematic_bounces = 4

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
