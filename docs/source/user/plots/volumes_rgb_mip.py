import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _volumes_rgb import SAMPLES, head, shoot, stage

import k3d


def generate():
    rgb, bounds = head()

    plot = stage()
    plot += k3d.mip(rgb, samples=SAMPLES, bounds=bounds)
    plot.camera = [180, 290, 110, 0, 0, 0, 0, 0, 1]

    return shoot(plot)
