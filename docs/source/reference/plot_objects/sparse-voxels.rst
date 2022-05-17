.. _sparse_voxels:

sparse_voxels
=============

.. autofunction:: k3d.factory.sparse_voxels

.. seealso::
    - :ref:`voxels`
    - :ref:`voxels_group`

Examples
--------

Basic
^^^^^

.. code-block:: python3

    import k3d
    import numpy as np

    sparse_voxels = np.array([[1, 0, 0, 1],
                              [0, 1, 0, 1],
                              [0, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 0, 2],
                              [0, 0, 1, 2],
                              [1, 0, 1, 2]]).astype(np.uint16)

    plt_sparse_voxels = k3d.sparse_voxels(sparse_voxels,
                                          space_size=[2, 2, 2],
                                          color_map=[0xfdc192, 0xa15525],
                                          outlines_color=0xffffff)

    plot = k3d.plot()
    plot += plt_sparse_voxels
    plot.display()

.. k3d_plot ::
  :filename: plots/sparse_voxels_basic_plot.py

Voxels to sparse voxels
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import numpy as np

    voxels = np.array([[[0, 1],
                        [1, 2]],
                       [[2, 2],
                        [1, 1]]])

    sparse_data = []

    for val in np.unique(voxels):
            if val != 0:
                z, y, x = np.where(voxels==val)
                sparse_data.append(np.dstack((x, y, z, np.full(x.shape, val))).reshape(-1,4).astype(np.uint16))

    sparse_voxels = np.vstack(sparse_data)

    """
    array([[1 0 0 1]
           [0 1 0 1]
           [0 1 1 1]
           [1 1 1 1]
           [1 1 0 2]
           [0 0 1 2]
           [1 0 1 2]])
    """