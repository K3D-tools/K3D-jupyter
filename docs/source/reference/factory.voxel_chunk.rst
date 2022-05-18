.. _voxel_chunk:

===========
voxel_chunk
===========

.. autofunction:: k3d.factory.voxel_chunk

.. seealso::
    - :ref:`voxels_group`

--------
Examples
--------

.. code-block:: python3

    import k3d
    import numpy as np

    voxels = np.array([[[0, 1],
                        [1, 2]],
                       [[2, 2],
                        [1, 1]]])

    chunk = k3d.voxel_chunk(voxels, [0, 0, 0])

    """
    VoxelChunk(coord=array([0, 0, 0], dtype=uint32),
               id=139644634801104,
               multiple=1,
               voxels=array([[[0, 1],
                              [1, 2]],
                             [[2, 2],
                              [1, 1]]], dtype=uint8))
    """