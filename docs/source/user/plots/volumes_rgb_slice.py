import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _volumes_rgb import head, shoot, stage

import k3d


def generate():
    rgb, bounds = head()
    nz, ny, nx = rgb.shape[:3]

    plot = stage()
    plot += k3d.volume_slice(rgb, slice_z=nz // 2, slice_y=ny // 2, slice_x=nx // 2,
                             bounds=bounds)
    plot.camera = [180, 290, 110, 0, 0, 0, 0, 0, 1]

    return shoot(plot)
