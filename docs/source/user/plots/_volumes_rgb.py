"""The Visible Human head, the page's figure for a volume that carries colour per voxel.

Data: visiblehuman.nii.gz from neurolabusc/niivue-images, an RGB24 NIfTI made from the Visible
Human Project cryosection photographs (U.S. National Library of Medicine). 196 x 240 x 256 voxels
at 1 mm, 36 MB once unpacked.

SimpleITK reads RGB24 straight into [z, y, x, 3] uint8 - the order k3d indexes a volume in - so
there is nothing to transpose. nibabel, which the example notebook uses, hands the same file over
as a structured dtype and needs a view first.

The array is cached at module level: the three scripts using it share one sphinx process.
"""
import os

import numpy as np
import SimpleITK as sitk

import k3d
from k3d.headless import get_headless_driver, k3d_remote

WIDTH = 1920
HEIGHT = 1080

# the ramp starts at 0.30, not just above the air. A volume is sampled trilinearly, so every
# surface has a rim where the texture fades in; a scalar field hides it, because the colormap
# turns whatever value the ray stops at into a full-intensity colour, while here the value IS the
# colour and a ray stopping halfway up the rim paints a half-bright one. Measured on this volume:
# the skin reads (70, 60, 46) with the ramp rising from 0.08 and (83, 70, 54) rising from 0.30.
OPACITY_FUNCTION = [0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0]
ALPHA_COEF = 120.0
SAMPLES = 512.0

_cache = {}


def head():
    """(rgb, bounds) - the volume in [z, y, x, 3] uint8 and its extent in millimetres."""
    if not _cache:
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../reference/assets/factory/visiblehuman.nii.gz')
        image = sitk.ReadImage(path)
        rgb = np.ascontiguousarray(sitk.GetArrayFromImage(image))

        dx, dy, dz = image.GetSpacing()
        nz, ny, nx = rgb.shape[:3]
        _cache['rgb'] = rgb
        _cache['bounds'] = [-nx * dx / 2, nx * dx / 2,
                            -ny * dy / 2, ny * dy / 2,
                            -nz * dz / 2, nz * dz / 2]

    return _cache['rgb'], _cache['bounds']


def stage():
    return k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0,
                    axes_helper=0,
                    screenshot_scale=1,
                    background_color=0,
                    menu_visibility=False)


def shoot(plot):
    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)
    png = headless.get_screenshot()
    headless.close()

    return png
