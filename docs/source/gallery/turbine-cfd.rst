Turbine CFD
===========

.. admonition:: References

    - :ref:`map_colors`
    - :ref:`paraview_color_maps`
    - :ref:`plot`
    - :ref:`vectors`
    - :ref:`vtk_poly_data`

:download:`cylinder.vtp <./assets/cylinder.vtp>`
:download:`cfd.csv <./assets/cfd.csv>`

.. code-block:: python3

    # Data and model from ParaView software examples

    import k3d
    import numpy as np
    import pandas as pd
    from k3d.helpers import map_colors
    from k3d.colormaps import paraview_color_maps
    from numpy.linalg import norm
    from vtk import vtkXMLPolyDataReader

    # Model
    reader = vtkXMLPolyDataReader()
    reader.SetFileName('cylinder.vtp')
    reader.Update()
    polydata = reader.GetOutput()

    plt_vtk = k3d.vtk_poly_data(polydata,
                                color=0x000000,
                                opacity=0.3)

    # CFD
    ccfd = pd.read_csv('cfd.csv')

    o = cfd[['Points:0', 'Points:1', 'Points:2']].values.astype(np.float32)
    v = cfd[['v:0', 'v:1', 'v:2']].values.astype(np.float32)

    norms = np.apply_along_axis(norm, 1, v.reshape(-1, 3))
    colors = map_colors(norms, paraview_color_maps.Cool_to_Warm_Extended).astype(np.uint32)
    colors = np.repeat(colors, 2)

    v = v * 100 / norm(v)

    plt_vectors = k3d.vectors(o, v,
                              head_size=0.5,
                              colors=colors,compression_level=7)

    # Plot
    plot = k3d.plot()
    plot += plt_vtk
    plot += plt_vectors
    plot.display()

    plot.camera = [3.0792, 14.6017, -8.8171,
                   -0.9959, 0.5287, -0.2337,
                   1, 0, 0]

.. k3d_plot ::
  :filename: plots/turbine_cfd_plot.py