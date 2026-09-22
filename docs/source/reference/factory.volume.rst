.. _volume:

======
volume
======
.. autofunction:: k3d.factory.volume

.. seealso::
    - :ref:`mip`

--------
Examples
--------

Render mhd volumetric data
^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`heart.mhd <./assets/factory/heart.mhd>`
:download:`heart.zraw <./assets/factory/heart.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    plt_volume = k3d.volume(img.astype(np.float32))

    plot = k3d.plot()
    plot += plt_volume
    plot.display()

.. k3d_plot ::
    :filename: plots/factory/volume_render_mhd_plot.py

Colormap
^^^^^^^^

:download:`heart.mhd <./assets/factory/heart.mhd>`
:download:`heart.zraw <./assets/factory/heart.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk
    from k3d.colormaps import matplotlib_color_maps

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    plt_volume = k3d.volume(img.astype(np.float32),
                            alpha_coef=250,
                            color_map=matplotlib_color_maps.Turbo,
                            color_range=[300, 900])

    plot = k3d.plot()
    plot += plt_volume
    plot.display()

.. k3d_plot ::
    :filename: plots/factory/volume_colormap_plot.py

Colour per voxel
^^^^^^^^^^^^^^^^

:download:`visiblehuman.nii.gz <./assets/factory/visiblehuman.nii.gz>`

A 4D array of ``uint8`` shaped ``[z, y, x, 3]`` or ``[z, y, x, 4]`` is colour measured per
voxel rather than a scalar to map, so it is drawn as it stands and ``color_map`` and
``color_range`` are refused with a warning. The alpha still has to come from somewhere: it is
Rec. 709 luminance shaped by ``opacity_function``, which is the whole transfer function here.

Start the ramp well above the empty space, not just above it. A volume is sampled trilinearly,
so every surface has a rim where the texture fades in, and a ray stopping halfway up that rim
paints a half-bright colour - a scalar field hides this because its colormap turns any value
into a full-intensity colour.

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    # RGB24 NIfTI: SimpleITK returns it as [z, y, x, 3] uint8, which is the order
    # a volume is indexed in, so there is nothing to transpose
    im_sitk = sitk.ReadImage('visiblehuman.nii.gz')
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    bounds = [-size[0] / 2, size[0] / 2,
              -size[1] / 2, size[1] / 2,
              -size[2] / 2, size[2] / 2]

    plt_volume = k3d.volume(img,
                            opacity_function=[0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0],
                            alpha_coef=120,
                            samples=512,
                            bounds=bounds)

    plot = k3d.plot(background_color=0, grid_visible=False)
    plot += plt_volume
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/volume_rgb_plot.py

Mask
^^^^

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

    plt_volume = k3d.volume(img.astype(np.float32),
                            mask=mask.astype(np.uint8),
                            mask_opacities=[0.025, 3.0],
                            color_range=[0, 700])

    plot = k3d.plot()
    plot += plt_volume
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/volume_mask_plot.py

Colour per voxel
^^^^^^^^^^^^^^^^

:download:`visiblehuman.nii.gz <./assets/factory/visiblehuman.nii.gz>`

A 4D array of ``uint8`` shaped ``[z, y, x, 3]`` or ``[z, y, x, 4]`` is colour measured per
voxel rather than a scalar to map, so it is drawn as it stands and ``color_map`` and
``color_range`` are refused with a warning. The alpha still has to come from somewhere: it is
Rec. 709 luminance shaped by ``opacity_function``, which is the whole transfer function here.

Start the ramp well above the empty space, not just above it. A volume is sampled trilinearly,
so every surface has a rim where the texture fades in, and a ray stopping halfway up that rim
paints a half-bright colour - a scalar field hides this because its colormap turns any value
into a full-intensity colour.

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    # RGB24 NIfTI: SimpleITK returns it as [z, y, x, 3] uint8, which is the order
    # a volume is indexed in, so there is nothing to transpose
    im_sitk = sitk.ReadImage('visiblehuman.nii.gz')
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    bounds = [-size[0] / 2, size[0] / 2,
              -size[1] / 2, size[1] / 2,
              -size[2] / 2, size[2] / 2]

    plt_volume = k3d.volume(img,
                            opacity_function=[0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0],
                            alpha_coef=120,
                            samples=512,
                            bounds=bounds)

    plot = k3d.plot(background_color=0, grid_visible=False)
    plot += plt_volume
    plot.display()

    plot.camera = [180, 290, 110, 0, 0, 0, 0, 0, 1]

.. k3d_plot ::
  :filename: plots/factory/volume_rgb_plot.py
