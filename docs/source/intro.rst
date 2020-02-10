Features
========

K3D-jupyter is a widget which primarily aims to be an easy and
efficient 3d visualization tool. It focuses on straightforward user
experience, providing similar API to well known matplotlib.  On the
other hand it uses the maximum of accelerated 3D graphics in the
browser. Those features make it a lightweight and efficient tool for
diversity of applications. Let give us some examples of use cases and
features of K3D-jupyter:

 - Photorealistic volume rendering for data on regular
   grids. Prominent example is Computer Tomography data, which can be
   visualized in real time with resolution of dataset
   :math:`512^3`. Moreover K3D-jupyter supports experimentally
   time-series which makes it possible to display 4D (3D + time)
   tomography in the browser.
 - 3d point plot - high performance renderer can display millions of
   points in a web browser. which is suitable for e.g. point cloud
   data coming from 3d scanners of for real time visualization of
   tracers particles in fluid dynamics.
 - Meshes with scalar attributes can be displayed, dynamically updated
   and animated.
 - Voxel geometry is supported on dense and sparse datasets. It is
   used for example in segmentation of CT data.


Interactivity with Ipywidgets
=============================

   
K3D-jupyter is an ipywidget, therefore it natively contains frontend
and backend. Backend is a Python process where the data is
prepared. Frontend is a javascript application with WebGL (via threejs
and custom pixelshaders) access. The ipywidget architecture allows for
communication of those two parts. K3D-jupyter exposes this
communication and allows for easy update of dataset on existing plot.

For example, if :code:`plt_points` is an :code:`k3d.points` object,
then a simple assignment on the backend (Python):

.. code::

   plt_points.positions = np.array([ [1,2,3],[3,2,1]], dtype=np.float32)

will trigger data transfer of the positions to the front-end and the
plot will be updated. 

One of the most attractive aspects of this architecture is the fact
that the backend process can run on arbitrary remote infrastructure,
e.g. in the HPC center, where large scale simulations are performed or
large datasets are available. Then is it possible to use the Jupyter
notebook as a kind of “remote display” for visualizing that data. As
the frontend is on user computer, the interactivity of inspection in
3d is very good, and also one can achieve fast updated of dataset or
its part.

Numpy first
===========

Similarly as in the case of matplotlib, the native data type is a numpy
array. It greatly simplifies usage of k3d and also simplifies the
frontend code as we implement only simple sets of objects. On the
other hand, the availability of the backend, does not prohibit from
using much more sophisticated visualization pipelines. An example
could be unstructured volumetric grid with some scalar field. K3D does
not support this kind of data, but it can be preprocessed using VTK
to - for example - mesh with color coded values. Such a mesh can be an
isosurface or section of the original volumetric data.  Moreover, if
such preprocessing will produce mesh which has small or moderate
number of triangles (e.g. :math:`<10^5`), then it can be interactively
explored by linking to other widgets.



Interactive snapshots
=====================

Plots can be saved as 'png' screenshots and interactive
html-snapshots, which contain the front-end javascript 3d application
bundles with the data. They can be used independently on the Python
back-end, send by email of embedd in webpages as iframe:


.. raw:: html

    <iframe src="_static/points.html" frameborder="0" height="300px" width="300px"></iframe>

Snapshots can be generated from Menu system or programatically:

.. code::
   
   plot.menu_visibility = False
   plot.fetch_snapshot()


In the latter case one has to write html code to a file:

   
.. code::
   
   with open('../_static/points.html','w') as fp:
       fp.write(plot.snapshot)

