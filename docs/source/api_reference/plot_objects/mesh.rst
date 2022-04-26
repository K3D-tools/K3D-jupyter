.. _mesh:

mesh
====

.. autofunction:: k3d.factory.mesh

.. seealso::
    - :ref:`surface`

**Examples**

Basic

.. code-block:: python3

    import k3d

    vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 3, 1], [3, 2, 1]]

    plt_tetra = k3d.mesh(vertices, indices,
                         colors=[0x32ff31, 0x37d3ff, 0xbc53ff, 0xffc700])

    plot = k3d.plot()
    plot += plt_tetra
    plot.display()

.. k3d_plot ::
  :filename: plots/mesh_basic_plot.py

Colormap

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps
    from matplotlib.tri import Triangulation

    n_radii = 8
    n_angles = 36

    radii = np.linspace(0.125, 1.0, n_radii, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False, dtype=np.float32)[..., np.newaxis]

    x = np.append(np.float32(0), (radii * np.cos(angles)).flatten())
    y = np.append(np.float32(0), (radii * np.sin(angles)).flatten())
    z = np.sin(-x * y)

    vertices = np.vstack([x, y, z]).T
    indices = Triangulation(x, y).triangles.astype(np.uint32)

    plt_mesh = k3d.mesh(vertices, indices,
                        color_map=matplotlib_color_maps.Jet,
                        attribute=z,
                        color_range=[-1.1, 2.01])

    plot = k3d.plot()
    plot += plt_mesh
    plot.display()

.. k3d_plot ::
  :filename: plots/mesh_colormap_plot.py