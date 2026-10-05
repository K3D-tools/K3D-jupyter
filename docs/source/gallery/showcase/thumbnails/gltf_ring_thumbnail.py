import importlib.util
import os

from k3d.headless import get_headless_driver, k3d_remote


def generate():
    # the same scene as the page, traced once at build time
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'gltf_ring_plot.py')
    spec = importlib.util.spec_from_file_location('gltf_ring_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plot = module.scene('cinematic')
    plot.screenshot_scale = 1
    plot.axes_helper = 0

    # a million triangles to build a BVH for before the first sample
    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800, refresh_timeout=1800)

    headless.sync(hold_until_refreshed=True)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
