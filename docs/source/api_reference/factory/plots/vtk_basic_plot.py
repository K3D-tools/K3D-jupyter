import k3d
import os
import vtk


def generate():
    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '../assets/cow.vtp')

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()

    plt_vtk = k3d.vtk_poly_data(polydata,
                                color=0xc6884b,
                                model_matrix=(1.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0, 0.0,
                                              0.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 1.0))

    plot = k3d.plot()
    plot += plt_vtk

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
