.. _factory.volume:

factory.volume
==============

.. autofunction:: k3d.factory.volume

.. seealso::
    - :ref:`factory.mip`

**Examples**

Render mhd volumetric data

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
  :filename: plots/volume_render_mhd_plot.py

Colormap

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
  :filename: plots/volume_colormap_plot.py