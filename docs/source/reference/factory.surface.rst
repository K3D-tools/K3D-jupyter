.. _surface:

=======
surface
=======

.. autofunction:: k3d.factory.surface

.. seealso::
    - :ref:`mesh`

--------
Examples
--------

Basic
^^^^^

.. code-block:: python3

    # Example from
    # https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/35311/versions/3/previews/html/Surface_Contour_Plot.html

    import k3d
    import numpy as np
    from numpy import sin, sqrt

    x = np.linspace(-5, 5, 100, dtype=np.float32)
    y = np.linspace(-5, 5, 100, dtype=np.float32)

    x, y = np.meshgrid(x, y)
    f = sin(sqrt(x**2 + y**2))

    plt_surface = k3d.surface(f,
                              color=0x006394,
                              wireframe=True,
                              xmin=0, xmax=10,
                              ymin=0, ymax=10)


    plot = k3d.plot()
    plot += plt_surface
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/surface_basic_plot.py

Colormap
^^^^^^^^

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    # Example from http://www2.math.umd.edu/~jmr/241/surfaces.html

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps

    x = np.linspace(-5, 5, 100, dtype=np.float32)
    y = np.linspace(-5, 5, 100, dtype=np.float32)

    x, y = np.meshgrid(x, y)
    f = ((x**2 - 1) * (y**2 - 4) + x**2 + y**2 - 5) / (x**2 + y**2 + 1)**2

    plt_surface = k3d.surface(f * 2,
                              xmin=-5, xmax=5,
                              ymin=-5, ymax=5,
                              compression_level=9,
                              color_map=matplotlib_color_maps.Coolwarm_r,
                              attribute=f, color_range=[-1, 0.5])


    plot = k3d.plot()
    plot += plt_surface
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/surface_colormap_plot.py