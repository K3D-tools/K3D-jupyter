.. _lines:

=====
lines
=====

.. autofunction:: k3d.factory.lines

--------
Examples
--------

Basic
^^^^^

:download:`cow.vtp <./assets/factory/cow.vtp>`

.. code-block:: python3

    # VTP model from https://github.com/naucoin/VTKData/blob/master/Data/cow.vtp

    import k3d
    import numpy as np
    import pyvista as pv

    data = pv.raed('cow.vtp')

    plt_vtk = k3d.vtk_poly_data(data)

    lines = k3d.lines(plt_vtk.vertices, plt_vtk.indices,
                  shader='mesh', width=0.025,
                  color=0xc6884b,
                  model_matrix=(1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += lines
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/lines_plot.py

