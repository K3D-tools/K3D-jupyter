.. _points:

======
points
======

.. autofunction:: k3d.factory.points

--------
Examples
--------

Basic
^^^^^

.. code-block:: python3

    import k3d
    import numpy as np

    x = np.random.randn(1000,3).astype(np.float32)

    plt_points = k3d.points(positions=x,
                            point_size=0.2,
                            shader='3d',
                            color=0x3f6bc5)

    plot = k3d.plot()
    plot += plt_points
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/points_basic_plot.py

Colormap
^^^^^^^^

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps

    x = np.random.randn(10000, 3).astype(np.float32)
    f = (np.sum(x ** 3 - .1 * x ** 2, axis=1))

    plt_points = k3d.points(positions=x,
                            point_size=0.1,
                            shader='flat',
                            opacity=0.7,
                            color_map=matplotlib_color_maps.Coolwarm,
                            attribute=f,
                            color_range=[-2, 1])

    plot = k3d.plot()
    plot += plt_points
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/points_colormap_plot.py