Visible Human
=============

.. admonition:: References

    - :ref:`volume`
    - :ref:`volume_slice`
    - :ref:`mip`

:download:`visiblehuman.nii.gz <../../reference/assets/factory/visiblehuman.nii.gz>`

Cryosection photographs of a frozen cadaver, not a density field: the colour here was
measured with a camera. A volume whose last axis is 3 or 4 is drawn as the colour it holds,
so there is no colormap in this scene and nothing to tune about the palette.

.. code-block:: python3

    import k3d
    import numpy as np
    import SimpleITK as sitk

    # RGB24 NIfTI. SimpleITK returns it as [z, y, x, 3] uint8, which is the order a volume
    # is indexed in, so there is nothing to transpose; nibabel hands the same file over as a
    # structured dtype and needs a view first
    im_sitk = sitk.ReadImage('visiblehuman.nii.gz')
    img = np.ascontiguousarray(sitk.GetArrayFromImage(im_sitk))
    size = np.array(im_sitk.GetSize()) * np.array(im_sitk.GetSpacing())

    plt_volume = k3d.volume(img,
                            opacity_function=[0.0, 0.0, 0.30, 0.0, 0.42, 1.0, 1.0, 1.0],
                            alpha_coef=120,
                            samples=512,
                            bounds=[-size[0] / 2, size[0] / 2,
                                    -size[1] / 2, size[1] / 2,
                                    -size[2] / 2, size[2] / 2])

    plot = k3d.plot(background_color=0,
                    grid_visible=False,
                    camera_auto_fit=False,
                    colorbar_object_id=0)
    plot += plt_volume
    plot.display()

    plot.camera = [180, 290, 110, 0, 0, 0, 0, 0, 1]

``opacity_function`` is the whole transfer function here, and it runs along Rec. 709
luminance, because colour has no other scalar to offer and the march has to know where to
stop. The same luminance feeds the gradient the shader lights with, which is why the skin
still catches a highlight.

Where the ramp rises is the one thing worth getting right. A volume is sampled trilinearly,
so every surface has a rim where the texture fades in; a scalar field hides it, because its
colormap turns whatever value the ray stops at into a full-intensity colour, while here the
value *is* the colour and a ray stopping halfway up the rim paints a half-bright one.
Rising from 0.30 rather than from just above the embedding medium puts the surface where
the skin is already itself: measured on this volume, that is the difference between
``(70, 60, 46)`` and ``(83, 70, 54)``.

The other two views of the same array need no settings at all. ``volume_slice`` cuts
photographs out of it, and ``mip`` keeps, for every ray, the colour of the brightest voxel
it passed - which on a head is bone and teeth, through the skin, in the colours they have.

.. k3d_plot ::
  :filename: plots/visible_human_plot.py
