import os

import numpy as np
import SimpleITK as sitk

import k3d
from k3d.headless import get_headless_driver, k3d_remote


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    plt_volume = k3d.volume(img.astype(np.float32))

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_volume

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
