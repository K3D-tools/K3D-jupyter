======
Basics
======

-------------
Create a plot
-------------

You can create a new plot using the :ref:`plot` function. |br|
Then use the ``display()`` method to show the plot below a Jupyter notebook <Jupyter>`_ cell.

.. code-block:: python3

    import k3d

    plot = k3d.plot()

    plot.display() # You can also just use 'plot'

If K3D-jupyter is properly installed, after executing the above snippet you
should see an empty plot:

.. k3d_plot::
  :filename: plots/empty_plot.py

.. note::
  In the above example, you had a ``Plot`` and added objects to
  it. It is also possible to automatically generate a plot for a
  created object:

  .. code-block:: python3

      import k3d

      k3d.points([0, 0, 0])

  However, this is not a good practice because a ``Plot`` object is created
  behind the scenes. If there are many of them, showing complex objects, a
  lot of browser memory will be used.

-------------------
Add objects to plot
-------------------

You can interactively add objects to a plot using the ``+=`` operator:

.. code:: ipython3

    import k3d

    plot = k3d.plot()

    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
    indices = [[0, 1, 2]]

    mesh = k3d.mesh(vertices, indices)

    plot += mesh

    plot.display()

.. k3d_plot ::
    :filename: plots/triangle_plot.py

And also add an object directly without creating a variable:

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
     :filename: plots/dual_triangle_plot.py

.. note::

    In this example, there are 2 displays of the plot associated with 2
    different cell outputs, however they are the same plot.

    In the Jupyter notebook, you should see the same scene (3 triangles) on both of them.
    Each view of the plot can be adjusted separately using the mouse.

In the same way, you can remove objects with the ``-=`` operator:

.. code:: python3

    plot -= mesh

Having variables then become convenient if you want to modify objects
already shown.

.. note::

    It is possible to automatically generate a plot for a
    created object, like:

    .. code:: python3

        import k3d

        k3d.points([0, 0, 0])

    However this is not a good practice, because a ``Plot`` object is created
    behind the scenes. If there are many of them, showing complex objects, a
    lot of browser memory will be used.

------------
Control menu
------------

The plot scene contains a :ref:`K3D panel <panel>` in its right top corner a foldable menu,
providing access to the most usefull plot options and listing all objects
you added to the scene.


View / camera position adjustment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plot can be adjusted using mouse actions which can be in one of
three modes: ``Trackball``, ``Orbit`` and ``Fly``.

The default ``Trackball`` mode works as follows:

- *mouse-wheel* ↦	 controls the zooming (in / out)
- *left-mouse* ↦	 drag rotates the plot (all directions)
- *right-mouse* ↦	 drag translates the plot (all directions)
- *mouse-wheel* ↦	 click and vertical drag controls the zooming (in / out)

To return to the default camera position, press the ``Reset camera`` button.

Fullscreen and detachted mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can switch the plot to fullscreen mode using the
``Fullscreen`` checkbox. To exit fullscreen mode, press the
``Esc`` key -- there should be a notification from your browser.

In a multiple monitor setup, it may be useful to detach the
plot to a dedicated window. This can be achieved by clicking
the ``Detach widget`` button.

Screenshots and snapshots
^^^^^^^^^^^^^^^^^^^^^^^^^

You can save a screenshot of the current view by pressing the ``Screenshot`` button.
The filename will be generated as "K3D-", followed by a decimal timestamp
and then ".png".

You can also make it programmatically using:

.. code:: python3

    plot.fetch_screenshot()

The PNG file is contained in the ``plot.screenshot`` attribute,
however, its synchronization might be a little bit delayed -- it relies
on an internal asynchronous traitlets_ mechanism.

Snapshot is a live version of a scene in the form of stand-alone
HTML file. Similarily to snapshots, you can either press the ``Snapshot HTML``
button or do it programmatically using:

.. code:: python3

    plot.get_snapshot()

In this case, you will have to write the ouput into an HTML file:

.. code:: python3

    with open('plot.html','w') as fp:
        fp.write(plot.get_snapshot())

------------
Plot options
------------

When you create a new plot using the :ref:`plot` function,
you can specify several options which control the behaviour and appearance of the
plot, such as:

-  ``height`` - the vertical size of the plot widget
-  ``antialias`` - enables antialiasing in the WebGL renderer, its
   effect depends on your WebGL implementation and browser settings.
-  ``background_color`` - RGB value of the background color packed into a
   single integer.

For example, to modify the background colour, you can do:

.. code:: ipython3

    plot.background_color = 0x00ffff

.. |br| raw:: html
    
    <br />

.. Links
.. _Jupyter: https://jupyter.org/
.. _traitlets: https://traitlets.readthedocs.io/en/stable/
