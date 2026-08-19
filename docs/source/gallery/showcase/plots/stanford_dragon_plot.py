import zipfile

import numpy as np
import vtk

import k3d
from k3d.helpers import download


def generate():
    filename = download(
        'https://casual-effects.com/g3d/data10/research/model/dragon/dragon.zip')

    with zipfile.ZipFile(filename) as archive:
        archive.extract('dragon.obj')

    reader = vtk.vtkOBJReader()
    reader.SetFileName('dragon.obj')

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())

    to_z_up = vtk.vtkTransform()
    to_z_up.RotateX(90)

    upright = vtk.vtkTransformPolyDataFilter()
    upright.SetTransform(to_z_up)
    upright.SetInputConnection(triangles.GetOutputPort())
    upright.Update()

    dragon = upright.GetOutput()

    bounds = np.array(dragon.GetBounds()).reshape(3, 2)
    centre = bounds.mean(axis=1)
    size = float((bounds[:, 1] - bounds[:, 0]).max())

    # half the thumbnail's budget: this one accumulates in the reader's browser
    plot = k3d.plot(renderer='cinematic',
                    environment='venice_sunset',
                    tone_mapping='aces',
                    grid_visible=False,
                    camera_auto_fit=False,
                    cinematic_samples=128,
                    cinematic_bounces=6,
                    # polished metal under a small bright sun throws fireflies:
                    # bright single pixels on the floor that no sample count clears
                    cinematic_glossy_filter=0.25)

    plot += k3d.vtk_poly_data(dragon,
                              color=0xC8A15A,
                              flat_shading=False,
                              roughness=0.2,
                              metalness=0.7,
                              compression_level=5,
                              name='dragon')

    floor_z = float(bounds[2, 0])
    span = 0.9 * size

    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] + span, floor_z],
                               [centre[0] - span, centre[1] + span, floor_z]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x9AA0A6,
                     roughness=0.9,
                     name='floor')

    # the head points along XY (0.75, 0.66): standing 28 degrees off that axis keeps it
    # turned towards the camera while the arc of the back and the coiled tail stay visible
    eye = centre + np.array([1.02, 0.245, 0.38]) * size
    plot.camera = [*eye, *centre, 0, 0, 1]

    plot.snapshot_type = 'inline'
    return plot.get_snapshot()
