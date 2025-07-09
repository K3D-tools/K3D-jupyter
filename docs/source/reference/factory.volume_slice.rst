.. _volume_slice:

============
volume_slice
============

.. autofunction:: k3d.factory.volume_slice

--------
Examples
--------

Render mhd volumetric data as three plane view
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`heart.mhd <./assets/factory/heart.mhd>`
:download:`heart.zraw <./assets/factory/heart.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    size = np.array(im_sitk.GetSpacing()) * np.array(im_sitk.GetSize())
    bounds = np.array([0, size[2], 0, size[1], 0, size[0]], np.float32)

    volume_slice = k3d.volume_slice(img.astype(np.float16),
                                  slice_z=img.shape[0]//2,
                                  slice_y=img.shape[1]//2,
                                  slice_x=img.shape[2]//2,
                                  bounds=bounds)

    plot = k3d.plot(camera_mode='volume_sides', grid_visible=False, background_color=0)
    plot.menu_visibility = False
    plot.slice_viewer_object_id = volume_slice.id

    plot += volume_slice
    plot.display()

.. k3d_plot ::
    :filename: plots/factory/volume_slice_plot.py

Mask
^^^^

Render mhd volumetric data as three plane view with mask viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`heart.mhd <./assets/factory/heart.mhd>`
:download:`heart.zraw <./assets/factory/heart.zraw>`
:download:`heart.mhd <./assets/factory/mask.mhd>`
:download:`heart.zraw <./assets/factory/mask.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    mask_sitk = sitk.ReadImage('mask.mhd')
    mask = sitk.GetArrayFromImage(mask_sitk)

    size = np.array(im_sitk.GetSpacing()) * np.array(im_sitk.GetSize())
    bounds = np.array([0, size[2], 0, size[1], 0, size[0]], np.float32)

    volume_slice = k3d.volume_slice(img.astype(np.float16),
                                    color_map=np.array(k3d.paraview_color_maps.Grayscale,
                                                       dtype=np.float32),
                                    slice_z=245,
                                    slice_y=199,
                                    slice_x=215,
                                    bounds=bounds,
                                    mask=mask.astype(np.uint8),
                                    active_masks= np.unique(mask)[1:],
                                    mask_opacity=0.9)

    plot = k3d.plot(camera_mode='volume_sides', grid_visible=False, background_color=0)
    plot.menu_visibility = False
    plot.slice_viewer_object_id = volume_slice.id
    plot += volume_slice
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/volume_slice_mask_plot.py

Outline mask
^^^^^^^^^^^^

Render mhd volumetric data as three plane view with outline mask viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`heart.mhd <./assets/factory/heart.mhd>`
:download:`heart.zraw <./assets/factory/heart.zraw>`
:download:`heart.mhd <./assets/factory/mask.mhd>`
:download:`heart.zraw <./assets/factory/mask.zraw>`

.. code-block:: python3

    import k3d
    from k3d.helpers import contour
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    mask_sitk = sitk.ReadImage('mask.mhd')
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
                                    color_map_masks=k3d.nice_colors,
                                    bounds=bounds,
                                    mask=mask.astype(np.uint8),
                                    active_masks= np.unique(mask)[1:],
                                    mask_opacity=0.75)

    plot = k3d.plot(camera_mode='volume_sides', grid_visible=False, background_color=0)
    plot.menu_visibility = False
    plot.slice_viewer_object_id = volume_slice.id
    plot += volume_slice
    plot += mesh
    plot.slice_viewer_mask_object_ids = [mesh.id]
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/volume_slice_outline_mask_plot.py