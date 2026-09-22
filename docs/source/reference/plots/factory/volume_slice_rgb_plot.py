import os

import numpy as np
import SimpleITK as sitk

import k3d


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/visiblehuman.nii.gz')

    im_sitk = sitk.ReadImage(filepath)
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    nz, ny, nx = img.shape[:3]

    plt_slice = k3d.volume_slice(img,
                                 slice_z=nz // 2,
                                 slice_y=ny // 2,
                                 slice_x=nx // 2,
                                 bounds=[-size[0] / 2, size[0] / 2,
                                         -size[1] / 2, size[1] / 2,
                                         -size[2] / 2, size[2] / 2])

    plot = k3d.plot(camera_mode='volume_sides', background_color=0, grid_visible=False)
    plot += plt_slice
    plot.camera = [180, 290, 110, 0, 0, 0, 0, 0, 1]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
