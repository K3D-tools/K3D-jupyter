.. _voxels:

voxels
======

.. autofunction:: k3d.factory.voxels

.. seealso::
    - :ref:`sparse_voxels`
    - :ref:`voxels_group`

**Examples**

Basic

.. code-block:: python3

    import k3d
    import numpy as np

    voxels = np.array([[[0, 1],
                        [1, 2]],
                       [[2, 2],
                        [1, 1]]]).astype(np.uint8)

    plt_voxels = k3d.voxels(voxels,
                            color_map=[0xfdc192, 0xa15525],
                            outlines_color=0xffffff)

    plot = k3d.plot()
    plot += plt_voxels
    plot.display()

.. k3d_plot ::
  :filename: plots/voxels_basic_plot.py

Shapes

.. code-block:: python3

    import k3d
    import numpy as np

    t = np.linspace(-1.5, 1.5, 50, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing='ij')
    R = 1
    r = 0.5

    eq_heart = (x**2 + (9/4 * y**2) + z**2 - 1)**3 - (x**2 * z**3) - (9/200 * y**2 * z**3)
    eq_torus = (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2 * (x**2 + y**2)

    voxels_heart = np.zeros_like(eq_heart).astype(np.uint8)
    voxels_torus = np.zeros_like(eq_torus).astype(np.uint8)
    voxels_heart[eq_heart < 0] = 1
    voxels_torus[eq_torus > 0] = 1

    plt_voxels_heart = k3d.voxels(voxels_heart,
                                  color_map=[0xbc4749],
                                  outlines=False,
                                  bounds=[-1.5, 1.5,-1.5, 1.5,-1.5, 1.5])

    plt_voxels_torus = k3d.voxels(voxels_torus,
                                  color_map=[0x3b60e4],
                                  outlines=False,
                                  wireframe=True,
                                  bounds=[-1.5, 1.5,-1.5, 1.5,-1.5, 1.5],
                                  translation=[0, 0, -3.5])

    plot = k3d.plot()
    plot += plt_voxels_heart
    plot += plt_voxels_torus
    plot.display()

.. k3d_plot ::
  :filename: plots/voxels_shapes_plot.py