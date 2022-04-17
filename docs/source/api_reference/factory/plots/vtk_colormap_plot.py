import k3d
import os
import vtk
from k3d.colormaps import matplotlib_color_maps


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/bunny.vtp')

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()

    plt_vtk = k3d.vtk_poly_data(polydata,
                                color_attribute=('Normals', 0, 1),
                                color_map=matplotlib_color_maps.Rainbow,
                                model_matrix=(1.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0,
                                              0.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += plt_vtk

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
