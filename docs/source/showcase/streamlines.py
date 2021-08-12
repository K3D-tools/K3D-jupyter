import numpy as np
import k3d
from k3d.headless import k3d_remote, get_headless_driver
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

    data = np.load(str(path) + '/assets/streamlines_data.npz')
    v = data['v']
    lines = data['lines']
    vertices = data['vertices']
    indices = data['indices']

    plt_streamlines = k3d.line(lines, attribute=v, width=0.00004,
                               color_map=k3d.matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5], shader='mesh')

    plt_mesh = k3d.mesh(vertices, indices, opacity=0.25, wireframe=True, color=0x0002)

    plot += plt_streamlines
    plot += plt_mesh

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(0.5)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
