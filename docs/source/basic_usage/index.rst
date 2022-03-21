Create a plot
=========================

In this example we will learn how to create and display K3D plots.

The main object is the plot, which can be created using the ``plot()``
fuction from the ``k3d`` module.

To show the plot below a Jupyter cell, we call its ``display()`` method.

.. code-block:: python3

    import k3d

    plot = k3d.plot()

    plot.display()

If K3D-jupyter is installed properly, after executing the above snippet you
should see an empty plot:

.. k3d_plot::
  :filename: empty_plot.py

.. note::
  In the above example we had a ``Plot`` and added objects to
  it. It is however possible to automatically generate a plot for a
  created object, like:

  .. code-block:: python3

      import k3d

      k3d.points([0, 0, 0])

  This is, however not a good practice, because a ``Plot`` object is created
  behind the scenes. If there are many of them, showing complex objects, a
  lot of browser memory will be used.

Add objects to plot
===================

The main idea is that plot objects are
interactively added to the plot using the ``+=`` operator:

.. code:: ipython3

    import k3d

    plot = k3d.plot()

    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
    indices = [[0, 1, 2]]

    mesh = k3d.mesh(vertices, indices)

    plot += mesh

    plot.display()

.. k3d_plot ::
  :filename: triangle_plot.py

It is also possible to add an object directly, without creating a variable:

.. code:: python3

    plot += k3d.mesh([0, 1, 1,
                      1, 1, 0,
                      1, 1, 1,

                      1, 2, 2,
                      1, 1, 1,
                      2, 1, 1],
                     [0, 1, 2, 3, 4, 5], color=0x00ff00)

    plot.display()

.. k3d_plot ::
  :filename: dual_triangle_plot.py

.. note::

    In this example there are 2 displays of the plot, associated with 2
    different cell outputs. However, they are the same plot.

    In the Jupyter notebook, you should see the same scene (3 triangles) on both of them.
    Each view of the plot can be adjusted separately using the mouse.

The same way, objects can be remove with the ``-=`` operator:

.. code:: python3

    plot -= mesh

Having variables then become convenient if we want to modify objects
already shown.

.. note::

    In the above example we had a ``Plot`` and added objects to it.
    It is however possible to automatically generate a plot for a
    created object, like:

    .. code:: python3

        import k3d

        k3d.points([0, 0, 0])

    However this is not a good practice, because a ``Plot`` object is created
    behind the scenes. If there are many of them, showing complex objects, a
    lot of your browser's memory will be used.


GUI
===

The plot scene contains in the right top corner a foldable menu. It
provides access to most usefull plot options and list all objects
which have beed added to the scene.


View / camera position adjustment
---------------------------------

The plot can be adjusted using mouse actions which can be in one of
three modes: "Trackball/Orbit/Fly".

The default Trackball mode works as following:

- *mouse-wheel* controls the zooming (in / out)
- *left-mouse* drag rotates the plot (all directions)
- *right-mouse* drag translates the plot (all directions)
- *mouse-wheel* click and vertical drag controls the zooming (in / out)

To return to the default camera position, press the *Reset camera* button.

Fullscreen and detachted mode
-----------------------------

It is possible to switch the plot to fullscreen mode using the
*Fullscreen* checkbox. To exit fullscreen mode press the
*Esc* key -- there should be a notification from the browser.

In multiple monitor setups, it may be useful to detach the
plot to a dedicated window. This can be achieved by clicking the *Detach
widget* button.

Screenshots and snapshots
=========================

To save a screenshot of the current view, press the *Screenshot* button.
The filename will be generated as "K3D-", then a string of digits
(technically: decimal timestamp) and then ".png".

Screenshots can be made programatically by:

.. code:: python3

    plot.fetch_screenshot()

The PNG file is contained in the `plot.screenshot` attribute,
however its synchronization might be a little bit delayed (it relies
on asynchronous traitlets mechanism internally)

Snapshot is a live version of a scene in the form of stand-alone
HTML file. Similarily to snapshots it can be done programatically via:

.. code:: python3

    plot.get_snapshot()

In this case, it has to be written into an HTML file:

.. code:: python3

    with open('plot.html','w') as fp:
        fp.write(plot.get_snapshot())

Plot options
============

The ``plot()`` function in ``k3d`` module creates a ``Plot`` object.
There are a several options, which control the behavior and apperance of the
plot, for example:

-  ``height`` - vertical size of the plot widget
-  ``antialias`` - enables antialiasing in the WebGL renderer, its
   effect depends on the WebGL implementation and browser settings. On
   by default.
-  ``background_color`` - RGB value of the backgound color packed into a
   single integer.

For example, to modify the background color, we have to do:

.. code:: ipython3

    plot.background_color = 0x00ffff

where `0x00ffff` stands for RGB value in hex.