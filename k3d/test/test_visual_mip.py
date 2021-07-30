import k3d
import numpy as np
import pytest
from .plot_compare import *
from k3d.helpers import download
import vtk
from vtk.util import numpy_support


def test_mip():
    prepare()

    filename = download('https://vedo.embl.es/examples/data/embryo.slc')
    reader = vtk.vtkSLCReader()
    reader.SetFileName(filename)
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.mip(volume_data, samples=128)

    pytest.plot += volume

    compare('mip')


def test_mip_opacity_function():
    prepare()

    filename = download('https://vedo.embl.es/examples/data/embryo.slc')
    reader = vtk.vtkSLCReader()
    reader.SetFileName(filename)
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    volume = k3d.mip(volume_data, opacity_function=[0, 0.0, 0.2, 0.5, 1, 1.0], samples=128)

    pytest.plot += volume

    compare('mip_opacity_function')
