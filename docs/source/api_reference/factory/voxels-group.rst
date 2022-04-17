.. _factory.voxels_group:

factory.voxels_group
====================

.. autofunction:: k3d.factory.voxels_group

**Examples**

Basic

.. code-block:: python3

    import k3d
    import numpy as np

    voxels = np.array([[[0, 1],
                        [1, 2]],
                      [[2, 2],
                        [1, 1]]])

    chunk1 = k3d.voxel_chunk(voxels, [0, 0, 0])
    chunk2 = k3d.voxel_chunk(voxels, [3, 3, 3])

    group = [{'voxels': chunk1['voxels'],
              'coord': chunk1['coord'],
              'multiple': chunk1['multiple']},
            {'voxels': chunk2['voxels'],
              'coord': chunk2['coord'],
              'multiple': chunk2['multiple']}]

    ids = [chunk1['id'], chunk2['id']]

    plt_voxels_group = k3d.voxels_group(space_size=[10, 10, 10],
                                        voxels_group=group,
                                        chunks_ids=ids)

    plot = k3d.plot()
    plot += plt_voxels_group
    plot.display()

.. k3d_plot ::
  :filename: plots/voxels_group_basic_plot.py