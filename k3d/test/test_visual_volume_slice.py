import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

import k3d

from .plot_compare import compare, prepare


def test_volume_slice():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("./test/assets/volume.vti")
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = (
        numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.float32)
    )

    volume = k3d.volume_slice(
        volume_data, slice_x=x // 2, slice_y=y // 2, slice_z=z // 2
    )

    pytest.plot += volume

    compare("volume_slice", modes=("simple", "advanced"))


def test_volume_slice_view_slice():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("./test/assets/volume.vti")
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = (
        numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.float32)
    )

    volume = k3d.volume_slice(
        volume_data, slice_x=x // 2, slice_y=y // 2, slice_z=z // 2
    )

    pytest.plot.camera_mode = "slice_viewer"
    pytest.plot.camera = [1, 1, 1, 0, 0, 0, 0, 0, 1]  # to force camera sync
    pytest.plot.grid_visible = False
    pytest.plot += volume

    pytest.plot.slice_viewer_object_id = volume.id

    pytest.plot.slice_viewer_direction = "z"
    volume.slice_x, volume.slice_y, volume.slice_z = -1, -1, z // 2
    compare("volume_slice_view_slice_z", modes=("simple", "advanced"))

    pytest.plot.slice_viewer_direction = "x"
    volume.slice_x, volume.slice_y, volume.slice_z = x // 2, -1, -1
    compare("volume_slice_view_slice_dynamic_x", modes=("simple", "advanced"))

    pytest.plot.slice_viewer_direction = "y"
    volume.slice_x, volume.slice_y, volume.slice_z = -1, y // 2, -1
    compare("volume_slice_view_slice_dynamic_y", modes=("simple", "advanced"))

    volume.slice_y = volume.slice_y - 20
    compare("volume_slice_view_slice_dynamic_y_position", modes=("simple", "advanced"))


def test_volume_slice_two_channels():
    """Two volumes and a 2D colormap: the bivariate path VolumeSlice.js has always had.

    The JS side keys the shader's TEXTURE_COUNT off config.volume.length, so this only renders
    as two channels if the trait carries a list all the way to the wire - a bare Array() would
    coerce it into one 4D array and draw a single channel with a mangled shape.
    """
    prepare()

    _z, _y, _x = np.mgrid[0:32, 0:32, 0:32]
    along_x = (_x / 31.0).astype(np.float32)
    along_y = (_y / 31.0).astype(np.float32)

    # a 2D colormap is a flat run of (x, y, r, g, b) control points forming a COMPLETE grid:
    # createCanvasGradient2d takes the unique x and y values and needs every pair present.
    # fmt: off
    color_map = np.array([
        0, 0, 0, 0, 0,      # both low  -> black
        1, 0, 1, 0, 0,      # x high    -> red
        0, 1, 0, 1, 0,      # y high    -> green
        1, 1, 1, 1, 0,      # both high -> yellow
    ], dtype=np.float32)
    # fmt: on

    volume = k3d.volume_slice(
        [along_x, along_y],
        color_map=color_map,
        color_range=[0, 1, 0, 1],
        slice_x=-1,
        slice_y=-1,
        slice_z=16,
    )

    pytest.plot.camera = [0, 0, 1.8, 0, 0, 0, 0, 1, 0]
    pytest.plot.grid_visible = False
    pytest.plot += volume

    compare("volume_slice_two_channels", modes=("simple", "advanced"))
