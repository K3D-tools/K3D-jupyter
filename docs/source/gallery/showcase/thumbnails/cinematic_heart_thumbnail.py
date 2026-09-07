import os

import numpy as np
import SimpleITK as sitk

import k3d
from k3d.headless import get_headless_driver, k3d_remote

# A cold build renders this in software, and it is the only cinematic thumbnail in the gallery
# that traces a CT volume rather than a mesh - at 256 that took minutes and looked like a hung
# build. 64 with the denoiser on is indistinguishable at the 155 px the grid shows.
SAMPLES = 64
BOUNCES = 5
WIDTH = 800
HEIGHT = 800

# the framing Artur settled on, read off the plot with the camera he orbited to
CAMERA = [82.32, -141.33, 74.31, -3.12, 5.48, -7.39, 0.03, 0.17, 0.99]
FOCUS_DISTANCE = 131.2

# that framing was set on the page's plot, which is wider than it is tall. A gallery thumbnail is
# square, so the same vertical fit crops the sides - back off along the same line of sight, which
# keeps the angle, the target and the sharp plane and only widens what fits.
PULL_BACK = 1.15


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())

    color_map = (np.array(k3d.colormaps.matplotlib_color_maps.OrRd_r).reshape(-1, 4)
                 * np.array([1, 1.25, 1.25, 1.25])).astype(np.float32)

    plt_volume = k3d.volume(img.astype(np.float16),
                            alpha_coef=250,
                            samples=256,
                            light_scale=2.25,
                            color_range=[300, 900],
                            color_map=color_map,
                            compression_level=5)

    plt_volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                                   -size[1] / 2, size[1] / 2,
                                   -size[2] / 2, size[2] / 2]

    plot = k3d.plot(renderer='cinematic',
                    environment='neutral',
                    background_color=0x2A2C30,
                    grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0,
                    axes_helper=0,
                    screenshot_scale=1,
                    cinematic_samples=SAMPLES,
                    cinematic_bounces=BOUNCES,
                    cinematic_denoise=2.0,
                    cinematic_bokeh_size=10.0,
                    lighting=1.5)
    plot += plt_volume

    position = np.array(CAMERA[0:3])
    target = np.array(CAMERA[3:6])
    offset = position - target

    plot.cinematic_focus_distance = FOCUS_DISTANCE + np.linalg.norm(offset) * (PULL_BACK - 1.0)
    plot.camera = [*(target + offset * PULL_BACK), *target, *CAMERA[6:9]]

    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)

    png = headless.get_screenshot()
    headless.close()

    return png
