.. _mip:

mip
===

.. autofunction:: k3d.factory.mip

.. seealso::
    - :ref:`volume`

Examples
--------

Render mhd volumetric data
^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`heart.mhd <./assets/heart.mhd>`
:download:`heart.zraw <./assets/heart.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    plt_mip = k3d.mip(img.astype(np.float32))

    plot = k3d.plot()
    plot += plt_mip
    plot.display()

.. k3d_plot ::
  :filename: plots/mip_render_mhd_plot.py

Colormap
^^^^^^^^

:download:`heart.mhd <./assets/heart.mhd>`
:download:`heart.zraw <./assets/heart.zraw>`

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk
    from k3d.colormaps import matplotlib_color_maps

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)

    plt_mip = k3d.mip(img.astype(np.float32),
                      color_map=matplotlib_color_maps.Turbo,
                      color_range=[100, 750])

    plot = k3d.plot()
    plot += plt_mip
    plot.display()

.. k3d_plot ::
  :filename: plots/mip_colormap_plot.py