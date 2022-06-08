K3D-jupyter Documentation
===========================

.. raw:: html

    <style>
        .grid-container {
            display: flex;
            column-gap: 20px;
            row-gap: 20px;
            flex-flow: row wrap;
        }

        .grid-container a {
            flex: 1 0 40%;
            max-width: 50%;
            max-height: 50%;
            border: 3px solid #343a40;
        }

        .grid-container2 img {
            width: 48%;
            vertical-align: bottom;
        }
    </style>

K3D-jupyter is a `Jupyter Notebook`_ 3D visualization package.

The primary aim of K3D-jupyter is to be an easy 3D visualization tool,
designed with native interoperation with existing powerful
libraries such as PyVista_, without being strictly dependent on them.

K3D-jupyter plots ipywidgets_ assuring a perfect interaction of a browser-side widget with Python kernel by a proven and standardized protocol.

--------------
Brief Examples
--------------

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d import matplotlib_color_maps

    data = np.load('streamlines_data.npz')
    plot = k3d.plot()
    plot += k3d.line(data['lines'], width=0.00007, color_range=[0, 0.5], shader='mesh',
                     attribute=data['v'], color_map=matplotlib_color_maps.Inferno)

    plot += k3d.mesh(data['vertices'], data['indices'],
                     opacity=0.25, wireframe=True, color=0x0002)
    plot.display()

.. k3d_plot ::
  :filename: plot.py

-----------------
Advanced Examples
-----------------

.. raw:: html

    <div class="grid-container2">

.. image:: ../../imgs/points_cloud.gif

.. image:: ../../imgs/streamlines.gif

.. image:: ../../imgs/vr.gif

.. image:: ../../imgs/tf_edit.gif

.. raw:: html

    </div>


-------------------
Offscreen rendering
-------------------

Click to open YouTube video.

.. raw:: html

    <div class="grid-container">

.. image:: https://i3.ytimg.com/vi/zCeQ_ZXy_Ps/maxresdefault.jpg
  :target: https://www.youtube.com/watch?v=zCeQ_ZXy_Ps

.. image:: https://i3.ytimg.com/vi/9evYSq3ieVs/maxresdefault.jpg
  :target: https://www.youtube.com/watch?v=9evYSq3ieVs

.. image:: https://i3.ytimg.com/vi/DbCiauTuJrU/maxresdefault.jpg
  :target: https://www.youtube.com/watch?v=DbCiauTuJrU

.. image:: https://i3.ytimg.com/vi/wIbBpUlB5vc/maxresdefault.jpg
  :target: https://www.youtube.com/watch?v=wIbBpUlB5vc

.. raw:: html

    </div>


.. toctree::
    :maxdepth: 1
    :hidden:

    User Guide <user/index>
    API reference <reference/index>
    Gallery <gallery/index>

.. Links
.. _Jupyter Notebook: https://jupyter.org/
.. _PyVista: https://docs.pyvista.org/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/

