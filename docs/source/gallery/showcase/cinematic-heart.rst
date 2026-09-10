Cinematic heart
===============

.. admonition:: References

    - :ref:`volume`
    - :ref:`plot`
    - :ref:`cinematic`

:download:`heart.mhd <../../reference/assets/factory/heart.mhd>`
:download:`heart.zraw <../../reference/assets/factory/heart.zraw>`

A cardiac CT as a participating medium rather than a ray march: light scatters inside the
tissue, the denser boundaries shade as surfaces, and what lights them is the environment.

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    im_sitk = sitk.ReadImage('heart.mhd')
    img = sitk.GetArrayFromImage(im_sitk)
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())

    # OrRd_r with its colour channels lifted above 1: the transfer function is what the
    # medium emits and reflects, so values over unity read as brighter tissue
    color_map = (np.array(k3d.colormaps.matplotlib_color_maps.OrRd_r).reshape(-1, 4)
                 * np.array([1, 1.25, 1.25, 1.25])).astype(np.float32)

    plt_volume = k3d.volume(img.astype(np.float16),
                            alpha_coef=250,
                            samples=256,
                            light_scale=2.25,
                            color_range=[300, 900],
                            color_map=color_map)

    plt_volume.transform.bounds = [-size[0] / 2, size[0] / 2,
                                   -size[1] / 2, size[1] / 2,
                                   -size[2] / 2, size[2] / 2]

    plot = k3d.plot(renderer='cinematic',
                    environment='neutral',
                    background_color=0x2A2C30,
                    grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0,
                    cinematic_samples=256,
                    cinematic_bounces=5,
                    cinematic_denoise=2.0,
                    cinematic_bokeh_size=10.0,
                    cinematic_focus_distance=185.2,
                    lighting=1.5)
    plot += plt_volume
    plot.display()

    # aimed at the centre of mass of what the colour range keeps, not at the centre of the
    # data box - the tissue is nowhere near it
    plot.camera = [-3.3, -221.6, 91.6, -3.3, 7.1, 17.3, 0, 0, 1]

The four knobs worth turning first. ``alpha_coef`` is how much matter a ray meets, and it
sets the cost as well as the density - lower it and light reaches deeper. ``light_scale``
is the medium's own exposure, separate from ``plot.lighting``, because a dense volume needs
more light than the geometry around it. ``color_range`` decides what counts as tissue at
all. And ``cinematic_bounces`` is what carries light into the interior: at 1 the inside
goes black however bright the environment.

The lens is the other half of the look. The tracer samples a real aperture, so depth of
field costs no extra samples - though a defocused frame wants more of them, because it
averages over a wider set of paths. ``cinematic_focus_distance`` is set here rather than
left at 0: zero focuses on whatever the camera is pointed at, which is the centre of the
data box, and that sits behind the surface anyone is actually looking at. The value here is
about three quarters of the camera's own distance, which puts the plane on the front of the
tissue and leaves the ribs behind it soft.

.. k3d_plot ::
  :filename: plots/cinematic_heart_plot.py
