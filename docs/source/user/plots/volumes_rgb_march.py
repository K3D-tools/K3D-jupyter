import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _volumes_rgb import ALPHA_COEF, OPACITY_FUNCTION, SAMPLES, head, shoot, stage

import k3d


def generate():
    rgb, bounds = head()

    plot = stage()
    plot += k3d.volume(rgb, opacity_function=OPACITY_FUNCTION, alpha_coef=ALPHA_COEF,
                       samples=SAMPLES, bounds=bounds)
    plot.camera = [227, -265, 132, 0, 0, 0, 0, 0, 1]

    return shoot(plot)
