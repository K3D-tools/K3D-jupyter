import os
import sys

import numpy as np

import k3d
from k3d.headless import get_headless_driver, k3d_remote

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots'))

from visible_human_plot import ALPHA_COEF, CAMERA, OPACITY_FUNCTION, SAMPLES, head

WIDTH = 800
HEIGHT = 800

# the page's framing was set on a plot wider than it is tall; a gallery thumbnail is square, so
# the same vertical fit crops the sides. Back off along the same line of sight, which keeps the
# angle and the target and only widens what fits.
PULL_BACK = 1.1


def generate():
    img, bounds = head()

    plt_volume = k3d.volume(img,
                            opacity_function=OPACITY_FUNCTION,
                            alpha_coef=ALPHA_COEF,
                            samples=SAMPLES,
                            bounds=bounds)

    plot = k3d.plot(background_color=0,
                    grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0,
                    axes_helper=0,
                    menu_visibility=False,
                    screenshot_scale=1)
    plot += plt_volume

    position = np.array(CAMERA[0:3])
    target = np.array(CAMERA[3:6])
    plot.camera = [*(target + (position - target) * PULL_BACK), *target, *CAMERA[6:9]]

    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)

    png = headless.get_screenshot()
    headless.close()

    return png
