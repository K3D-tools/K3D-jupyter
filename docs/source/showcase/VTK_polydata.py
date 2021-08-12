import k3d
from k3d.headless import k3d_remote, get_headless_driver
import pathlib
import vtk

path = pathlib.Path(__file__).parent.resolve()


def generate():
    plot = k3d.plot(screenshot_scale=1.0)
    headless = k3d_remote(plot, get_headless_driver(), width=320, height=226)

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

    headless.sync(hold_until_refreshed=True)
    headless.camera_reset(1.0)

    screenshot = headless.get_screenshot()
    headless.close()

    return screenshot
