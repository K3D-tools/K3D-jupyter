import k3d
import numpy as np
import pytest
from .plot_compare import *
import vtk
from vtk.util import numpy_support


def test_mip():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('./test/assets/volume.vti')
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.mip(volume_data, samples=512)

    pytest.plot += volume

    compare('mip')


def test_mip_opacity_function():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('./test/assets/volume.vti')
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.mip(volume_data, opacity_function=[0, 0.0, 0.2, 0.5, 1, 1.0], samples=512)

    pytest.plot += volume

    compare('mip_opacity_function')
