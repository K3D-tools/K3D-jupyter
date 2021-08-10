import numpy as np
import k3d
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, camera_auto_fit=False)

    data = np.load(str(path) + '/assets/streamlines_data.npz')
    v = data['v']
    lines = data['lines']
    vertices = data['vertices']
    indices = data['indices']

    plt_streamlines = k3d.line(lines, attribute=v, width=0.00007,
                               color_map=k3d.matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5], shader='mesh')

    plt_mesh = k3d.mesh(vertices, indices, opacity=0.25, wireframe=True, color=0x0002)

    plot.camera = [0.064, 0.043, 0.043, 0.051, 0.041, 0.049, -0.059, 0.993, 0.087]
    plot += plt_streamlines
    plot += plt_mesh

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
