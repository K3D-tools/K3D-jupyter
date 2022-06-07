import k3d
import numpy as np
import os
import SimpleITK as sitk
from k3d.colormaps import matplotlib_color_maps


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    plt_mip = k3d.mip(img.astype(np.float32),
                      color_map=matplotlib_color_maps.Viridis,
                      color_range=[100, 750])

    plot = k3d.plot()
    plot += plt_mip

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
