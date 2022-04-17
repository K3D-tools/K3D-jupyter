import k3d
import numpy as np
import os
import SimpleITK as sitk
from k3d.colormaps import matplotlib_color_maps


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    plt_volume = k3d.volume(img.astype(np.float32),
                            alpha_coef=250,
                            color_map=matplotlib_color_maps.Turbo,
                            color_range=[300, 900])

    plot = k3d.plot()
    plot += plt_volume

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
