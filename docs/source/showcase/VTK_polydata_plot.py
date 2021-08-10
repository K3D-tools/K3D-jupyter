import k3d
import pathlib
import vtk

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0)

    model_matrix = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path) + '/assets/cow.vtp')
    reader.Update()

    cow3d = k3d.vtk_poly_data(reader.GetOutput(), color=0xff0000,
                              model_matrix=model_matrix)
    plot += cow3d

    plot.snapshot_type = 'inline'

    return plot.get_snapshot()
