"""One cardiac CT, rendered once per renderer for the volume comparison on the renderers page.

The array is cached at module level: the three scripts using it share one sphinx process, so the
19 MB file is read and padded once.

One thing to know about what is and is not held constant. alpha_coef, the transfer function, the
camera and the environment are identical, so every difference in shape and shading is the renderer.
Exposure is not: light_scale is a uniform the raster volume shader never reads - grep
Volume.fragment.glsl - so 2.25 lifts the traced image only. That is deliberate. The march lights
every sample locally and counts the environment twice while the tracer attenuates, so at this
density an unlifted traced image is far darker, and the comparison would be about brightness
instead of about light. Set LIGHT_SCALE to 1.0 to see that raw difference.
"""
import os
import time

import numpy as np

import k3d
from k3d.headless import get_headless_driver, k3d_remote

# full HD: these are the page's reference figures for what the renderers do to a volume,
# and the difference between advanced and cinematic lives in detail a thumbnail loses
WIDTH = 1920
HEIGHT = 1080

# the framing the showcase settled on, so the two pages read as the same subject
CAMERA = [82.32, -141.33, 74.31, -3.12, 5.48, -7.39, 0.03, 0.17, 0.99]
FOCUS_DISTANCE = 131.2
LIGHT_SCALE = 2.25
ALPHA_COEF = 250
SAMPLES = 128

# advanced only - the occlusion pass is what it adds over simple, and its defaults (0.07 / 1.8)
# are tuned for geometry. A volume contributes the shell where its accumulated opacity crosses
# one half, and on a chest that shell is finely branched, so a wide radius and a gentle exponent
# read better than a tight, deep one.
AO_RADIUS = 0.1
AO_STRENGTH = 0.5

_scan = None


def _heart():
    global _scan

    if _scan is not None:
        return _scan

    import SimpleITK as sitk

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../reference/assets/factory/heart.mhd')
    image = sitk.ReadImage(filepath)

    _scan = (np.asarray(sitk.GetArrayFromImage(image)),
             np.array(image.GetSize()) * np.array(image.GetSpacing()))

    return _scan


def screenshot(renderer, shadow='off'):
    img, size = _heart()

    color_map = (np.array(k3d.colormaps.matplotlib_color_maps.OrRd_r).reshape(-1, 4)
                 * np.array([1, 1.25, 1.25, 1.25])).astype(np.float32)

    volume = k3d.volume(img.astype(np.float16),
                        alpha_coef=ALPHA_COEF,
                        samples=256,
                        light_scale=LIGHT_SCALE,
                        color_range=[300, 900],
                        color_map=color_map,
                        shadow=shadow,
                        compression_level=5)

    volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                               -size[1] / 2, size[1] / 2,
                               -size[2] / 2, size[2] / 2]

    plot = k3d.plot(renderer=renderer,
                    environment='neutral',
                    background_color=0x2A2C30,
                    grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0,
                    axes_helper=0,
                    screenshot_scale=1,
                    ao_radius=AO_RADIUS,
                    ao_strength=AO_STRENGTH,
                    lighting=1.5)

    if renderer == 'cinematic':
        plot.cinematic_samples = SAMPLES
        plot.cinematic_bounces = 5
        plot.cinematic_denoise = 2.0
        plot.cinematic_bokeh_size = 10.0
        plot.cinematic_focus_distance = FOCUS_DISTANCE

    plot += volume
    plot.camera = CAMERA

    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)

    # the light map is built on BEFORE_RENDER, shadow_delay after the scene settles, so the
    # first frame of an on-demand shadow is drawn without it - spend one and wait it out
    if shadow != 'off':
        headless.get_screenshot()
        time.sleep(2.0)

    png = headless.get_screenshot()
    headless.close()

    return png
