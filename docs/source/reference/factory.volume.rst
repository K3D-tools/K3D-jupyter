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

