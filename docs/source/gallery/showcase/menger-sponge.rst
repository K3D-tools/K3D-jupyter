Menger sponge
=============

.. admonition:: References

    - :ref:`voxels`
    - :ref:`plot`

.. code-block:: python3

    import k3d
    import numpy as np

    def iterate(length, x, y, z):
        nl = length // 3

        if nl < 1:
            return

        margin = (nl - 1) // 2

        voxels[z - margin:z + margin + 1, y - margin:y + margin + 1, :] = 0
        voxels[z - margin:z + margin + 1, :, x - margin:x + margin + 1] = 0
        voxels[:, y - margin:y + margin + 1, x - margin:x + margin + 1] = 0

        for ix, iy, iz in np.ndindex((3, 3, 3)):
            if (1 if ix != 1 else 0) + (1 if iy != 1 else 0) + (1 if iz != 1 else 0) != 2:
                iterate(nl, x + (ix - 1) * nl, y + (iy - 1) * nl, z + (iz - 1) * nl)

    iteration = 4
    size = 3 ** iteration

    voxels = np.ones((size, size, size))

    iterate(size, size // 2, size // 2, size // 2)

    plt_voxels = k3d.voxels(voxels.astype(np.uint8),
                            color_map=(0xfdfe03), outlines=True)

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False)
    plot += plt_voxels
    plot.display()

    plot.camera = [98.5152, -60.0912, 88.9902,
                   43.4731, 37.6014, 31.5219,
                   -0.2226, 0.3405, 0.9135]

.. k3d_plot ::
  :filename: plots/menger_sponge_plot.py