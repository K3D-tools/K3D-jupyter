import numpy as np

import k3d


def generate():
    g = np.linspace(-1, 1, 64, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing='ij')
    blob = (np.exp(-(x ** 2 + y ** 2 + z ** 2) / 0.35) * 900).astype(np.float32)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    screenshot_scale=1.0,
                    colorbar_object_id=0,
                    renderer='advanced')

    plot += k3d.volume(blob, samples=256, alpha_coef=120,
                       color_map=k3d.matplotlib_color_maps.jet,
                       color_range=[80, 900],
                       compression_level=7)

    plot += k3d.mesh(np.array([[-1.3, -1.3, -0.5], [1.3, -1.3, 0.4],
                               [1.3, 1.3, 0.4], [-1.3, 1.3, -0.5]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0xD9C089, side='double', name='plane')

    plot.depth_peels = 4
    plot.camera = [2.4, -2.4, 1.6, 0, 0, 0, 0, 0, 1]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
