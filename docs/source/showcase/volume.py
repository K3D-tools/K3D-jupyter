import numpy as np
import k3d
from k3d.headless import k3d_remote, get_headless_driver
import SimpleITK as sitk
import pathlib

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0, lighting=2.0)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

    im_sitk = sitk.ReadImage(str(path) + '/assets/heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    im_sitk.GetSize()

    volume = k3d.volume(
        img.astype(np.float32),
        alpha_coef=1000,
        shadow='dynamic',
        samples=600,
        shadow_res=128,
        shadow_delay=50,
        color_range=[150, 750],
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.Gist_heat).reshape(-1, 4)
                   * np.array([1, 1.75, 1.75, 1.75])).astype(np.float32)
    )

    volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                               -size[1] / 2, size[1] / 2,
                               -size[2] / 2, size[2] / 2]
    plot += volume

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
