.. _name:

=============
Object naming
=============

When you add an object to a plot, the :ref:`K3D panel <panel>` creates a new submenu in
the `Objects` section and attributes to it a default name made with the object type,
followed by its order of addition to the plot. |br|
For example, `Mesh #2` for the second ``mesh`` added to the plot.

.. k3d_plot ::
  :filename: plots/object_naming_unnamed_plot.py

Let's suppose with the above example that you want to hide the green and yellow point sets. |br|
The object not being labelled, it will be difficult to find the right one at first sight.

To overcome this issue and improve the interactivity, you can assign to each plot object a custom
name using the ``name`` attributes. |br|
As a result, your objects will be listed using those names.

.. code-block:: python3
    
    import k3d
    import numpy as np

    xs = []
    for i in range(5):
        xs.append(np.random.randn(150, 3).astype(np.float32))

    plt_points_red = k3d.points(xs[0],
                                point_size=0.2,
                                color=0xff0000,
                                name='Red data')
    plt_points_green = k3d.points(xs[1] - 0.9,
                                  point_size=0.2,
                                  color=0x00ff00,
                                  name='Green data')
    plt_points_blue = k3d.points(xs[2] + 0.5,
                                 point_size=0.2,
                                 color=0x0000ff,
                                 name='Blue data')
    plt_points_yellow = k3d.points(xs[3] - 1.5,
                                   point_size=0.2,
                                   color=0xffff00,
                                   name='Yellow data')
    plt_points_black = k3d.points(xs[4] + 1,
                                  point_size=0.2,
                                  color=0x000000,
                                  name='Black data')

    plot = k3d.plot(name='Multidata point cloud')
    plot += plt_points_red
    plot += plt_points_green
    plot += plt_points_blue
    plot += plt_points_yellow
    plot += plt_points_black
    plot.display()

.. k3d_plot ::
    :filename: plots/object_naming_named_plot.py

.. |br| raw:: html
    
    <br />