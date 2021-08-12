VTK polydata
============

.. code::

    import numpy as np
    import k3d
    import vtk

    plot = k3d.plot()

    model_matrix = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName('/assets/cow.vtp')
    reader.Update()

    cow3d = k3d.vtk_poly_data(reader.GetOutput(), color=0xff0000,
                              model_matrix=model_matrix)
    plot += cow3d

    plot.display()

.. k3d_plot ::
   :filename: VTK_polydata_plot.py