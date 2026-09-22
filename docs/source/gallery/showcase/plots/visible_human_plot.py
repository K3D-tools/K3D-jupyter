import os

import numpy as np
import SimpleITK as sitk

import k3d

# the ramp starts at 0.30 rather than just above the embedding medium. A volume is sampled
# trilinearly, so every surface has a rim where the texture fades in, and with colour per voxel
# a ray stopping halfway up that rim paints a half-bright colour: the skin reads (70, 60, 46)
# with the ramp rising from 0.08 and (83, 70, 54) rising from 0.30
OPACITY_FUNCTION = [0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0]
ALPHA_COEF = 120
SAMPLES = 512
# the file is RAS, so -y is the back of the head; the face is on +y
CAMERA = [180, 290, 110, 0, 0, 0, 0, 0, 1]


def head():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../../reference/assets/factory/visiblehuman.nii.gz')

    im_sitk = sitk.ReadImage(filepath)
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())

    return img, [-size[0] / 2, size[0] / 2,
                 -size[1] / 2, size[1] / 2,
                 -size[2] / 2, size[2] / 2]


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
                    screenshot_scale=1)
    plot += plt_volume
    plot.camera = CAMERA

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
