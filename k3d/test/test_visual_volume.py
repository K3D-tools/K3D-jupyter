import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

import k3d

from .plot_compare import compare, prepare


def test_volume():
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

    volume = k3d.volume(volume_data, samples=128)

    pytest.plot += volume

    compare("volume")


def test_volume_opacity_function():
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

    volume = k3d.volume(
        volume_data, opacity_function=[0, 0.0, 0.2, 0.5, 1, 1.0], samples=128
    )

    pytest.plot += volume

    compare("volume_opacity_function")


def test_volume_alpha_coef():
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

    volume = k3d.volume(volume_data, alpha_coef=200, samples=128)

    pytest.plot += volume

    compare("volume_alpha_coef")


def test_volume_mask():
    prepare()

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName("./test/assets/heart.mhd")
    reader.Update()
    vti = reader.GetOutput()

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName("./test/assets/mask.mhd")
    reader.Update()
    mask = reader.GetOutput()

    bounds = vti.GetBounds()

    x, y, z = vti.GetDimensions()
    volume_data = (
        numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.float32)
    )

    x, y, z = mask.GetDimensions()
    mask_data = (
        numpy_support.vtk_to_numpy(mask.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.uint8)
    )

    volume = k3d.volume(
        volume_data,
        mask=mask_data,
        mask_opacities=[1.0, 1.0],
        color_range=[0, 700],
        alpha_coef=200,
        samples=128,
        bounds=bounds,
    )

    pytest.plot += volume

    compare("volume_heart")

    volume.mask_opacities = [0.025, 1.0]

    compare("volume_heart_dynamic_mask_opacities")


def test_volume_mesh_depth_peel_composition():
    """A mesh intersecting a volume interleaves correctly when depth peeling is
    on (issue #277): the volume march is split into segments bounded by the peel
    layer depths. Contract: depth_peels >= 3; below that the segmentation is too
    coarse to be predictable."""
    prepare(depth_peels=4)

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

    pytest.plot += k3d.volume(volume_data, samples=256)
    pytest.plot += k3d.mesh(
        np.array(
            [[-0.45, -0.45, -0.3], [0.45, -0.45, 0.1],
             [0.45, 0.45, 0.1], [-0.45, 0.45, -0.3]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        color=0x22AA44,
        side="double",
    )

    compare("volume_mesh_depth_peels")
