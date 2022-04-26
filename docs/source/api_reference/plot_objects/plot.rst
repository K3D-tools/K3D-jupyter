.. _plot:

plot
====

.. autofunction:: k3d.factory.plot

**Examples**

Coloring

.. code-block:: python3

    import k3d

    plot = k3d.plot(background_color=0x1e1e1e,
                    grid_color=0xd2d2d2,
                    label_color=0xf0f0f0)

    plot.display()

.. k3d_plot ::
  :filename: plots/plot_coloring_plot.py

Axes and bounds

.. code-block:: python3

    import k3d

    plot = k3d.plot(grid=(0, 0, 0, 60, 20, 50),
                    axes=['Time', 'Mass', 'Temperature'])

    plot.display()

.. k3d_plot ::
  :filename: plots/plot_axes_and_bounds_plot.py

Static

.. code-block:: python3

    import k3d

    plot = k3d.plot(camera_no_pan=True,
                    camera_no_rotate=True,
                    camera_no_zoom=True,
                    menu_visibility=False)

    plot.display()

.. k3d_plot ::
  :filename: plots/plot_static_plot.py