Volume rendering
================

.. code::

    import numpy as np
    import k3d
    import SimpleITK as sitk

    plot = k3d.plot()

    im_sitk = sitk.ReadImage('./assets/heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())
    im_sitk.GetSize()

    volume = k3d.volume(
        img.astype(np.float32),
        alpha_coef=1000,
        shadow='dynamic',
        samples=600,
        shadow_res=128,
        shadow_delay=50,
        color_range=[150, 750],
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.Gist_heat).reshape(-1, 4)
                   * np.array([1, 1.75, 1.75, 1.75])).astype(np.float32)
    )

    volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                               -size[1] / 2, size[1] / 2,
                               -size[2] / 2, size[2] / 2]
    plot.display()

.. k3d_plot ::
   :filename: volume_plot.py