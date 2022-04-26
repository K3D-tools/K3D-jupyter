.. _line:

line
=====

.. autofunction:: k3d.factory.line

**Examples**

Basic

.. code-block:: python3

    import k3d
    import numpy as np

    t = np.linspace(-10, 10, 100,dtype=np.float32)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5

    vertices = np.vstack([x,y,z]).T

    plt_line = k3d.line(vertices, width=0.1, color=0xff99cc)

    plot = k3d.plot()
    plot += plt_line
    plot.display()

.. k3d_plot ::
  :filename: plots/line_basic_plot.py

Colormap

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps

    t = np.linspace(-10, 10, 100,dtype=np.float32)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5

    vertices = np.vstack([x,y,z]).T

    plt_line = k3d.line(vertices, width=0.2, shader='mesh',
                    color_map=matplotlib_color_maps.Jet,
                    attribute=t,
                    color_range=[-5, 5])

    plot = k3d.plot()
    plot += plt_line
    plot.display()

.. k3d_plot ::
  :filename: plots/line_colormap_plot.py