import os

import numpy as np
import SimpleITK as sitk

import k3d
from k3d.headless import get_headless_driver, k3d_remote


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/visiblehuman.nii.gz')

    im_sitk = sitk.ReadImage(filepath)
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())

    plt_volume = k3d.volume(img,
                            opacity_function=[0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0],
                            alpha_coef=120,
                            samples=512,
                            bounds=[-size[0] / 2, size[0] / 2,
                                    -size[1] / 2, size[1] / 2,
                                    -size[2] / 2, size[2] / 2])

    plot = k3d.plot(screenshot_scale=1,
                    background_color=0,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_volume

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
