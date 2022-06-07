Turbine CFD
===========

.. admonition:: References

    - :ref:`map_colors`
    - :ref:`paraview_color_maps`
    - :ref:`plot`
    - :ref:`vectors`
    - :ref:`vtk_poly_data`

:download:`cylinder.vtp <./assets/cylinder.vtp>`
:download:`cfd.npz <./assets/cfd.npz>`

.. code-block:: python3

    # Data and model from ParaView software examples

    import k3d
    import numpy as np
    import pyvista as pv
    from k3d.helpers import map_colors
    from k3d.colormaps import paraview_color_maps
    from numpy.linalg import norm

    cylinder = pv.read('cylinder.vtp')
    plt_vtk = k3d.vtk_poly_data(cylinder,
                                color=0x000000,
                                opacity=0.3)

    data = np.load('cfd.npz')
    plt_vectors = k3d.vectors(data['o'], data['v'] * 3,
                              line_width=0.02,
                              colors=data['c'])

    plot = k3d.plot()
    plot += plt_vtk
    plot += plt_vectors
    plot.display()

    plot.camera = [3.0792, 14.6017, -8.8171,
                   -0.9959, 0.5287, -0.2337,
                   1, 0, 0]

.. k3d_plot ::
  :filename: plots/turbine_cfd_plot.py