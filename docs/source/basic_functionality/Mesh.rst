Mesh
====

Mesh is an object that displays triangles in 3d. A scalar can be displayed on the mesh
using color map.

.. code::

    # this code is a part of matplotlib trisurf3d_demo

    import numpy as np
    import k3d
    from matplotlib.tri import Triangulation

    plot = k3d.plot()

    n_radii = 8
    n_angles = 36

    radii = np.linspace(0.125, 1.0, n_radii, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False, dtype=np.float32)[..., np.newaxis]

    x = np.append(np.float32(0), (radii * np.cos(angles)).flatten())
    y = np.append(np.float32(0), (radii * np.sin(angles)).flatten())

    z = np.sin(-x * y)
    indices = Triangulation(x, y).triangles.astype(np.uint32)

    plt_mesh = k3d.mesh(np.vstack([x, y, z]).T, indices,
                        color_map=k3d.colormaps.basic_color_maps.Jet,
                        attribute=z,
                        color_range=[-1.1, 2.01]
                        )
    plot += plt_mesh
    plot.display()

.. k3d_plot ::
   :filename: mesh/Mesh01.py

Scalars can be updated interactively using ipywidgets communication:

.. code::

    plt_mesh.attribute = 2 * x ** 2 + y ** 2

.. k3d_plot ::
   :filename: mesh/Mesh02.py

It is possible to send a time series consisting of attribute values:

.. code::

    plt_mesh.attribute = {str(t): 3 * t * x ** 2 + y ** 2 for t in np.linspace(0, 2, 20)}
    plot.start_auto_play()

.. k3d_plot ::
   :filename: mesh/Mesh03.py

