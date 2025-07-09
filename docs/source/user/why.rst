======================
Why using K3D-jupyter?
======================

K3D-jupyter primarily aims is to be an easy and efficient 3D visualization tool. |br|
To do so, it focuses on straightforward user experience,
providing an API similar to the well-known matplotlib_,
and uses the maximum of accelerated 3D graphics in web browsers.

--------
Features
--------

K3D-jupyter includes many features, making it a lightweight and efficient tool for a
diverse range of applications. |br|
Some examples of use cases could be:

 - Photorealistic volume rendering for data on regular grids.

   A prominent example is `Computer Tomography`_ data, which can be
   visualized in real-time with a dataset resolution of :math:`512^3`.
   Moreover, it experimentally supports time series,
   making it possible to display 4D (3D + time) tomography_.

 - High-performance 3D point plot renderer able to display millions of
   points in web browsers.

   For example, point cloud data coming from 3D scanners or for
   real-time visualization of tracer particles in fluid dynamics.

 - Display meshes with scalar attributes with dynamic update
   and animation.

 - Voxel geometry for both dense and sparse datasets.
 
   For example, in `Computer Tomography`_  data segmentation.

-----------------------------
Interactivity with Ipywidgets
-----------------------------

K3D-jupyter being an ipywidgets_, it natively contains front-end
and back-end. |br|
The back-end is a Python process where the data is prepared, and
the frontend is a JavaScript application with WebGL_ (via `Three.js`_ and
custom `pixel shaders <pixelshaders>`_) access.

The ipywidgets_ architecture allows for communication of these two parts.
K3D-jupyter exposes this communication and allows for easy dataset updates on existing plots. |br|
For example, if :code:`plt_points` is an :ref:`points` object,
then a simple assignment in the backend

.. code-block:: python3

    plt_points.positions = np.array([[1,2,3],[3,2,1]], dtype=np.float32)

will trigger the data transfer of points positions to the front-end and update the plot.

One of the most attractive aspects of this architecture is
that the backend process can run on arbitrary remote infrastructure,
like in HPC_ centers where large scale simulations are performed or
large datasets are available.

Then it is possible to use the `Jupyter Notebook`_ as a remote display for visualizing that data.
As the frontend is on the users' computer, the interactivity of 3D
inspection is very good and can achieve fast updates on the whole
dataset.

-----------
NumPy first
-----------

As matplotlib_, the native data type is a NumPy_ array.

It greatly simplifies K3D-jupyter usage as well as the frontend
code since it implements only simple sets of objects.
On the other hand, the availability of the back-end does not prohibit from using
much more sophisticated visualization pipelines.

An example could be an unstructured volumetric grid with some scalar field.
K3D does not support this kind of data, but it can be preprocessed using PyVista_ to --
for example -- mesh it with color coded values.
Moreover, if such preprocessing produced a mesh with a small or moderate number of
triangles (:math:`<10^5`), then it could be interactively explored by
linking to other widgets.

---------------------
Interactive snapshots
---------------------

Plots can be saved as both PNG screenshots and interactive HTML
snapshots, which contain the front-end JavaScript 3D application
bundles with the data. They can be used independently on the Python
back-end, send by email or embedded in webpages.

.. |br| raw:: html

   <br />

.. Links
.. _matplotlib: https://matplotlib.org/
.. _tomography: https://en.wikipedia.org/wiki/Tomography
.. _Computer Tomography: https://en.wikipedia.org/wiki/CT_scan
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/
.. _WebGL: https://www.khronos.org/webgl/
.. _Three.js: https://threejs.org/
.. _HPC: https://en.wikipedia.org/wiki/High-performance_computing
.. _pixelshaders: https://www.nvidia.com/en-us/drivers/feature-pixelshader/
.. _NumPy: https://numpy.org
.. _PyVista: https://docs.pyvista.org/
.. _Jupyter Notebook: https://jupyter.org/