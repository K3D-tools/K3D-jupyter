Basic plottting  
===============

Creating and displaying a plot
------------------------------

In this example we will learn how to create and display K3D plots.

The main object is the plot, which can be created using the ``plot()``
fuction from the ``k3d`` module.

To show the plot below a Jupyter cell, we call its ``display()`` method.

.. code:: ipython3

    import k3d
    plot = k3d.plot()
    
    # here you would normally create objects to display
    # and add them to the plot
    
    plot.display()

If K3D-jupyter is installed properly, after executing the above snippet you
should see something like this:

.. k3d_plot ::
   :filename: basic_plotting/empty_plot.py
   :screenshot:

.. figure:: empty_plot.png
   :alt: Empty K3D plot
   :width: 540px
   :align: center
   :figclass: align-center

   An empty plot


In the next example we will learn how to display objects on the plot.


.. note:: In the above example we had a ``Plot`` and added objects to
    it. It is however possible to automatically generate a plot for a
    created object, like:

    .. code:: ipython3

              import k3d
    
              k3d.points([0, 0, 0])

    This is, however not a good practice, because a ``Plot`` object is created
    behind the scenes. If there are many of them, showing complex objects, a
    lot of your browser’s memory will be used.



Adding objects to plot
----------------------

The main idea is that plot objects (e.g. lines, meshed wtc) are
interactively added to the plot. 

To draw the triangle we will use the ``mesh()`` method from the ``k3d``
module. This method creates a ``Mesh`` object, which can be added to a
K3D ``Plot``.

.. code:: ipython3

    import k3d
    plot = k3d.plot()
    
    
    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
    indices = [[0, 1, 2]]
    
    mesh = k3d.mesh(vertices, indices)
    
    plot += mesh
    
    plot.display()

.. k3d_plot ::
   :filename: basic_plotting/basic_plotting_plot01.py
   :screenshot:

.. figure:: basic_plotting_plot01.png
   :alt: An isosceles triangle in the y=0 plane
   :width: 540px
   :align: center
   :figclass: align-center

   An isosceles triangle in the y=0 plane

The arguments we passed to the ``mesh()`` function are a vertex array (a
``list`` or NumPy’s ``ndarray`` is OK) which is composed of
:math:`(x, y, z)` coordinates and an array of index triplets
(``int``\ s). Each triplet refers to the vertex array, defining one
triangle.

We can of course add objects directly to the plot, without creating
variables:


.. code:: ipython3

    plot += k3d.mesh([0, 1, 1, 
                     1, 1, 0, 
                     1, 1, 1,
                     
                     1, 2, 2,
                     1, 1, 1,
                     2, 1, 1], [0, 1, 2, 3, 4, 5], color=0x00ff00)
    
    plot

.. k3d_plot ::
   :filename: basic_plotting/basic_plotting_plot02.py
   :screenshot:

.. figure:: basic_plotting_plot02.png
   :alt: An isosceles triangle in the y=0 plane
   :width: 540px
   :align: center
   :figclass: align-center

   One blue and two green triangles
   


This is a plot of two meshes. Please note – in the second case we didn’t
nest the triplets - the numbers run continuously in a flat list. We also
used an optional argument, ``color`` to specify the color of the second
object. K3D objects have many attributes, which we can find out about
from the docstrings and from other examples, dedicated to the specific
object type.

Back to the main topic. The ``plot`` keeps track of the objects that it
contains:

.. code:: ipython3

    len(plot.objects)

We have 2 displays of the plot in the notebook, associated with 2
different cell outputs. However, they are the same plot - you should see
the same scene (3 triangles) on both of them. Each view of the plot can
be adjusted separately using the mouse.

When the plot becomes too cluttered with objects, we may want to remove
some of them. This is easily done with the ``-=`` operator. This is the
place, where having named our objects beforehand comes in handy:

.. code:: ipython3

    plot -= mesh
    plot



Having variables is also convenient when we want to modify the objects
already shown. 




GUI Basics
----------

The plot scene contains in the right top corner a foldable menu. It
provides access to most usefull plot options and list all objects
which have beed added to the scene.


View / camera position adjustment
+++++++++++++++++++++++++++++++++

The plot can be adjusted using mouse actions which can be in one of
three modes: "Trackball/Orbit/Fly".  The default Trackball mode works
as following:
- mouse wheel / scroll controls the zooming in or out - dragging with
left mouse button rotates the plot (all directions) - dragging with
right mouse button translates the plot (all directions) - dragging
with wheel / both mose buttons: zooms in or out (only vertical)

To return to the default camera position, press the “Camera reset” icon
from the top-right toolbar

Fullscreen mode and detachted mode
++++++++++++++++++++++++++++++++++

It is possible to switch the plot to fullscreen mode using the
“Fullscreen” icon from the toolbar. To exit fullscreen mode press the
Esc key (there should be a notification from your browser).

Especially in multiple monitor setups it may be useful to detach the
plot to a dedicated window. This is achieved by clicking the “Detach
widget” icon.


.. _snapshots:
 
Screenshots and snapshots
-------------------------

To save a screenshot of the current view, press the “Save screenshot”
icon from the toolbar. It provides better resolution, which can be
controlled by `plot.screenshot_scale` parameter.

The filename will be generated as “K3D-”, then a string of digits
(technically: decimal timestamp) and then “.png”.

.. note: If the `plot.name` is set it will be used as a name of the screenshot.
   

Screenshots can be made programatically by:

.. code:: ipython3

    plot.fetch_screenshot()

The ".png" file is contained in the `plot.screenshot` attribute,
however its synchronization might be a little bit delayed (it relies
on asynchronous traitlets mechanism internally)


Snapshot is a "live" version of a screne in the form of stand-alone
html file. Similarily to snapshots it can be done programatically via:


   
 - on the javascript side `plot.fetch_snapshot()`, note that fetching
   might take some time, and `plot.snapshot`
 - on the python side `plot.get_snapshot()`
   
In this  case one has to write HTML code to a file:

   
.. code::
   
   with open('../_static/points.html','w') as fp:
       fp.write(plot.snapshot)

       



Plot options
------------

The ``plot()`` function in ``k3d`` module creates a ``Plot`` object.
There are a several options, which control the behavior and apperance of the
plot, for example:

-  ``height`` - vertical size of the plot widget
-  ``antialias`` - enables antialiasing in the WebGL renderer, its
   effect depends on the WebGL implementation and browser settings. On
   by default.
-  ``background_color`` - RGB value of the backgound color packed into a
   single integer.

For example to change the background we have to do:

.. code:: ipython3

    plot.background_color = 0x00ffff

where `0x00ffff` stands for RGB value in hex.

