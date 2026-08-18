"""Shared scene for the renderers guide: a roughness/metalness sweep over a floor.

Loaded by the plot scripts via a sys.path insert (the k3d_plot directive imports
each script file in isolation), so this module keeps to numpy + k3d.
"""
import numpy as np

import k3d

GOLD = 0xC28439
N = 6
STEP = 1.2
SPAN = (N - 1) * STEP


def material_grid_plot(**plot_kwargs):
    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    screenshot_scale=1.0,
                    colorbar_object_id=0,
                    **plot_kwargs)

    # bottom row dielectrics, top row metals; roughness sweeps left to right.
    # one object per sphere - roughness/metalness are per-object materials
    for j, metal in enumerate((0.0, 1.0)):
        for i in range(N):
            rough = 0.05 + 0.9 * i / (N - 1)
            plot += k3d.points(np.array([[i * STEP, j * 1.7, 0.55]], np.float32),
                               shader='mesh', point_size=1.0, mesh_detail=3,
                               color=GOLD, roughness=rough, metalness=metal,
                               name='rough %.2f metal %.0f' % (rough, metal))

    plot += k3d.mesh(np.array([[-1.4, -1.6, 0], [SPAN + 1.4, -1.6, 0],
                               [SPAN + 1.4, 3.3, 0], [-1.4, 3.3, 0]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x8A8F98, roughness=0.85, name='floor')

    plot.camera = [SPAN / 2, -7.4, 5.4,
                   SPAN / 2, 0.85, 0.2,
                   0, 0, 1]

    return plot
