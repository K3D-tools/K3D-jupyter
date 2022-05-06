Points
======

To draw points one needs to prepare data as an array of coordinates `[number_of_points, 3]`.
Colors for all the points can either be the same or have an individual value (`colors` attribute).

When the number of points is larger than $10^3$ it is recommended to use fast shaders: `flat`, `
3d` or `3dSpecular`. The `mesh` shader generates much bigger overhead, but it has a properly
triangularized sphere representing each point.

.. code::

    import k3d
    import numpy as np

    x = np.random.randn(1000,3).astype(np.float32)
    point_size = 0.2

    plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=x, point_size=0.2)
    plot += plt_points
    plt_points.shader='3d'
    plot.display()

.. k3d_plot ::
   :filename: points/Points01.py

We can color points with some scalar value:

.. code::

    f = (np.sum(x ** 3 - .1 * x ** 2, axis=1))
    colormap = k3d.colormaps.basic_color_maps.WarmCool
    colors = k3d.helpers.map_colors(f, colormap, [-2, .1])
    colors = colors.astype(np.uint32)
    plt_points.colors = colors

.. k3d_plot ::
   :filename: points/Points02.py