import os

import numpy as np
import SimpleITK as sitk

import k3d
from k3d.headless import get_headless_driver, k3d_remote

# the directive caches the PNG, so only a cold build pays for this
SAMPLES = 256
BOUNCES = 5
WIDTH = 800
HEIGHT = 800

# straight at the front of the chest, lifted 18 degrees: the ribs then stand edge-on at
# the margins instead of sweeping across the heart
VIEW = np.array([0.0, -1.0, 0.325])
DISTANCE = 1.3
# the colour range's lower bound, which also decides what counts as tissue for the aim
LOW = 300
# the sharp plane, as a fraction of the camera's own distance, so it stays on the front of the
# tissue when DISTANCE moves. 0 would focus on the camera's target instead, and that sits behind
# the surface anyone is actually looking at
FOCUS = 0.77


def centre_of_tissue(img, size, low):
    """World-space centroid of the voxels the colour range keeps.

    Marginals rather than argwhere: this is 83 million voxels and only three numbers are wanted.
    """
    mass = img > low
    # int64 throughout: numpy's default integer is 32-bit on Windows, and the first moment of
    # five million voxels over five hundred slices overflows it into a negative centroid
    total = float(mass.sum(dtype=np.int64))

    if total == 0.0:
        return np.zeros(3)

    # the array is (z, y, x) and bounds are (x, y, z)
    axes = [mass.sum(axis=(0, 1), dtype=np.int64),
            mass.sum(axis=(0, 2), dtype=np.int64),
            mass.sum(axis=(1, 2), dtype=np.int64)]
    centre = []

    for extent, counts in zip(size, axes):
        index = float((counts * np.arange(counts.shape[0], dtype=np.int64)).sum()) / total

        centre.append(-extent / 2.0 + (index + 0.5) / counts.shape[0] * extent)

    return np.array(centre)


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
                            color_range=[LOW, 900],
                            color_map=color_map,
                            compression_level=5)

    plt_volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                                   -size[1] / 2, size[1] / 2,
                                   -size[2] / 2, size[2] / 2]

    # aim at the centre of mass of what the colour range keeps, not at the centre of the data
    # box: the tissue is nowhere near it, and pointing at the box leaves the subject off frame
    target = centre_of_tissue(img, size, LOW)

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

    distance = DISTANCE * float(size.max())
    plot.cinematic_focus_distance = FOCUS * distance

    offset = VIEW / np.linalg.norm(VIEW) * distance
    plot.camera = [*(target + offset), *target, 0, 0, 1]

    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)

    png = headless.get_screenshot()
    headless.close()

    return png
