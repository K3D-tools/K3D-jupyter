import SimpleITK as sitk
import numpy as np
import os

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    plt_mip = k3d.mip(img.astype(np.float32))

    plot = k3d.plot()
    plot += plt_mip

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
