import SimpleITK as sitk
import numpy as np
import os

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/mask.mhd')

    mask_sitk = sitk.ReadImage(filepath)
    mask = sitk.GetArrayFromImage(mask_sitk)

    plt_volume = k3d.volume(img.astype(np.float32),
                            mask=mask.astype(np.uint8),
                            mask_opacities=[0.025, 3.0],
                            color_range=[0, 700])

    plot = k3d.plot()
    plot += plt_volume

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
