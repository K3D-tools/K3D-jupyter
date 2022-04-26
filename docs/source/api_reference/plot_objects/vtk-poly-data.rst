.. _vtk_poly_data:

vtk_poly_data
=============

.. autofunction:: k3d.factory.vtk_poly_data

**Examples**

Basic

:download:`cow.vtp <./assets/cow.vtp>`

.. code-block:: python3

    # VTP model from https://github.com/naucoin/VTKData/blob/master/Data/cow.vtp

    import k3d
    import numpy as np
    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName('cow.vtp')
    reader.Update()
    polydata = reader.GetOutput()

    plt_vtk = k3d.vtk_poly_data(polydata,
                                color=0xc6884b,
                                model_matrix = (1.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 1.0, 0.0,
                                                0.0, 1.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += plt_vtk
    plot.display()

.. k3d_plot ::
  :filename: plots/vtk_basic_plot.py

Colormap

:download:`bunny.vtp <./assets/bunny.vtp>`

.. attention::
    `color_map` must be used along with `color_attribute` in order to work correctly.

.. code-block:: python3

    # VTP model from https://github.com/pyvista/vtk-data/blob/master/Data/Bunny.vtp

    import k3d
    import vtk
    from k3d.colormaps import matplotlib_color_maps

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName('bunny.vtp')
    reader.Update()
    polydata = reader.GetOutput()

    plt_vtk = k3d.vtk_poly_data(polydata,
                                color_attribute=('Normals', 0, 1),
                                color_map=matplotlib_color_maps.Rainbow,
                                model_matrix = (1.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 1.0, 0.0,
                                                0.0, 1.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += plt_vtk
    plot.display()

.. k3d_plot ::
  :filename: plots/vtk_colormap_plot.py