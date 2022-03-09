Features
========

K3D-jupyter is a widget which primarily aims to be an easy and
efficient 3D visualization tool. It focuses on straightforward user
experience, providing API similar to well known Matplotlib. On the
other hand it uses the maximum of accelerated 3D graphics in the
browser. Those features make it a lightweight and efficient tool for
diversity of applications. Some examples of the use cases and features
of K3D-jupyter:

 - Photorealistic volume rendering for data on regular grids.
   Prominent example is Computer Tomography data, which can be
   visualized in real time with resolution of dataset :math:`512^3`.
   Moreover K3D-jupyter experimentally supports time-series, which
   makes it possible to display 4D (3D + time) tomography in the
   browser.
 - 3D point plot - high performance renderer can display millions of
   points in a web browser, which is suitable for e.g. point cloud
   data coming from 3D scanners or for real time visualization of
   tracers particles in fluid dynamics.
 - Meshes with scalar attributes can be displayed, dynamically updated
   and animated.
 - Voxel geometry is supported on both dense and sparse datasets. It is
   used for example in segmentation of CT data.

Interactivity with Ipywidgets
=============================

K3D-jupyter is an :code:`ipywidget`, therefore it natively contains frontend
and backend. Backend is a Python process where the data is prepared.
Frontend is a JavaScript application with WebGL (via Three.js and
custom pixelshaders) access. The :code:`ipywidget` architecture allows for
communication of these two parts. K3D-jupyter exposes this
communication and allows for easy dataset updates on existing plot.

For example, if :code:`plt_points` is an :code:`k3d.points` object,
then a simple assignment on the backend:

.. code-block:: python3

    plt_points.positions = np.array([[1,2,3],[3,2,1]], dtype=np.float32)

will trigger data transfer of the positions to the front-end and the
plot will be updated.

One of the most attractive aspects of this architecture is the fact
that the backend process can run on arbitrary remote infrastructure,
e.g. in the HPC center, where large scale simulations are performed or
large datasets are available. Then is it possible to use the Jupyter
notebook as a kind of “remote display” for visualizing that data.
As the frontend is on user's computer, the interactivity of 3D
inspection is very good. One can achieve fast updates on the whole
dataset or any of its parts.

Numpy first
===========

Similarly as in the case of matplotlib, the native data type is a
:code:`numpy` array. It greatly simplifies K3D usage as well as the frontend
code since we implement only simple sets of objects. On the other
hand, the availability of the backend does not prohibit from using
much more sophisticated visualization pipelines. An example could be
an unstructured volumetric grid with some scalar field. K3D does not
support this kind of data, but it can be preprocessed using VTK to -
for example - mesh with color coded values. Such a mesh can be an
isosurface or section of the original volumetric data. Moreover, if
such preprocessing produced a mesh with small or moderate number of
triangles (e.g. :math:`<10^5`), then it could be interactively explored by
linking to other widgets.

Interactive snapshots
=====================

Plots can be saved as both PNG screenshots and interactive HTML
snapshots, which contain the front-end JavaScript 3D application
bundles with the data. They can be used independently on the Python
back-end, send by email or embedded in webpages.
