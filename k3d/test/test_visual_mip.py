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

def test_mip_mask():
    prepare()

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName('./test/assets/heart.mhd')
    reader.Update()
    vti = reader.GetOutput()

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName('./test/assets/mask.mhd')
    reader.Update()
    mask = reader.GetOutput()

    bounds = vti.GetBounds()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    x, y, z = mask.GetDimensions()
    mask_data = numpy_support.vtk_to_numpy(
        mask.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.uint8)

    volume = k3d.mip(volume_data, mask=mask_data, mask_opacities=[1.0, 1.0],
                        color_range=[0, 700], alpha_coef=200, samples=128, bounds=bounds)

    pytest.plot += volume

    compare('mip_heart')

    volume.mask_opacities = [0.025, 1.0]

    compare('mip_heart_dynamic_mask_opacities')
