import SimpleITK as sitk
import numpy as np
import os

import k3d
from k3d.helpers import contour


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/heart.mhd')

    im_sitk = sitk.ReadImage(filepath)
    img = sitk.GetArrayFromImage(im_sitk)

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../../assets/factory/mask.mhd')

    mask_sitk = sitk.ReadImage(filepath)
    mask = sitk.GetArrayFromImage(mask_sitk)

    size = np.array(im_sitk.GetSpacing()) * np.array(im_sitk.GetSize())
    bounds = np.array([0, size[2], 0, size[1], 0, size[0]], np.float32)

    mesh = k3d.vtk_poly_data(contour(mask, bounds, [1]), color=k3d.nice_colors[1])

    volume_slice = k3d.volume_slice(img.astype(np.float16),
                                    color_map=np.array(k3d.paraview_color_maps.Grayscale,
                                                       dtype=np.float32),
                                    slice_z=245,
                                    slice_y=199,
                                    slice_x=215,
                                    bounds=bounds,
                                    mask=mask.astype(np.uint8),
                                    active_masks=np.unique(mask)[1:],
                                    color_map_masks=k3d.nice_colors,
                                    mask_opacity=0.75)

    plot = k3d.plot(camera_mode='volume_sides', grid_visible=False, background_color=0)
    plot.menu_visibility = False
    plot.slice_viewer_object_id = volume_slice.id
    plot += volume_slice
    plot += mesh
    plot.slice_viewer_mask_object_ids = [mesh.id]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
