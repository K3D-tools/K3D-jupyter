import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

import k3d
from .plot_compare import prepare, compare


def test_volume_slice():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('./test/assets/volume.vti')
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.volume_slice(volume_data, slice_x=x // 2, slice_y=y // 2, slice_z=z // 2)

    pytest.plot += volume

    compare('volume_slice')


def test_volume_slice_view_slice():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('./test/assets/volume.vti')
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.volume_slice(volume_data, slice_x=x // 2, slice_y=y // 2, slice_z=z // 2)

    pytest.plot.camera_mode = 'slice_viewer'
    pytest.plot.camera = [1, 1, 1, 0, 0, 0, 0, 0, 1]  # to force camera sync
    pytest.plot.grid_visible = False
    pytest.plot += volume

    pytest.plot.slice_viewer_object_id = volume.id

    pytest.plot.slice_viewer_direction = 'z'
    volume.slice_x, volume.slice_y, volume.slice_z = -1, -1, z // 2
    compare('volume_slice_view_slice_z')

    pytest.plot.slice_viewer_direction = 'x'
    volume.slice_x, volume.slice_y, volume.slice_z = x // 2, -1, -1
    compare('volume_slice_view_slice_dynamic_x')

    pytest.plot.slice_viewer_direction = 'y'
    volume.slice_x, volume.slice_y, volume.slice_z = -1, y // 2, -1
    compare('volume_slice_view_slice_dynamic_y')

    volume.slice_y = volume.slice_y - 20
    compare('volume_slice_view_slice_dynamic_y_position')
