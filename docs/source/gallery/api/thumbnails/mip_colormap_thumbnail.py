import k3d
import numpy as np
import os
import SimpleITK as sitk
from k3d.colormaps import matplotlib_color_maps
from k3d.headless import k3d_remote, get_headless_driver


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    plt_mip = k3d.mip(img.astype(np.float32),
                      color_map=matplotlib_color_maps.Viridis,
                      color_range=[100, 750])

    plot = k3d.plot(screenshot_scale=1,
                    grid_visible=False,
                    axes_helper=0)
    plot += plt_mip

    headless = k3d_remote(plot, get_headless_driver(), width=800, height=800)

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
