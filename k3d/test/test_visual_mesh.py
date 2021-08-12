import k3d
import numpy as np
import pytest
from .plot_compare import *
from k3d.helpers import download
import vtk
from vtk.util import numpy_support

vertices = [
    -10, 0, -1,
    10, 0, -1,
    10, 0, 1,
    -10, 0, 1,
]

indices = [
    0, 1, 3,
    1, 2, 3
]


def test_mesh():
    global vertices, indices

    prepare()

    mesh = k3d.mesh(vertices, indices)
    pytest.plot += mesh

    compare('mesh')


def test_mesh_attribute():
    global vertices, indices

    prepare()

    vertex_attribute = [0, 1, 1, 0]
    mesh = k3d.mesh(vertices, indices, attribute=vertex_attribute,
                    color_map=k3d.basic_color_maps.CoolWarm, color_range=[0.0, 1.0])
    pytest.plot += mesh

    compare('mesh_attribute')


def test_mesh_advanced():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(reader.GetOutput(), color=0x222222, flat_shading=True,
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))
    pytest.plot += mesh

    compare('mesh_advanced')


def test_mesh_advanced_smoothed():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(reader.GetOutput(), color=0x222222, flat_shading=False,
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))
    pytest.plot += mesh

    compare('mesh_advanced_smoothed')


def test_mesh_advanced_opacity():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(reader.GetOutput(), color=0x222222, flat_shading=False, opacity=0.5,
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))
    pytest.plot += mesh

    compare('mesh_advanced_opacity')


def test_mesh_advanced_wireframe():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(reader.GetOutput(), color=0x222222, opacity=0.2, wireframe=True,
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))
    pytest.plot += mesh

    compare('mesh_advanced_wireframe')


def test_mesh_attribute_advanced():
    prepare()

    N = 100

    theta = np.linspace(0, 2.0 * np.pi, N)
    phi = np.linspace(0, 2.0 * np.pi, N)
    theta, phi = np.meshgrid(theta, phi)

    c, a = 2, 1
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    vertices = np.dstack([x, y, z]).astype(np.float32)
    indices = (np.stack([
        np.arange(N * N) + 0, np.arange(N * N) + N, np.arange(N * N) + N + 1,
        np.arange(N * N) + 0, np.arange(N * N) + N + 1, np.arange(N * N) + 1
    ]).T % (N * N)).astype(np.uint32)

    mesh = k3d.mesh(vertices, indices, flat_shading=False,
                    attribute=phi,
                    transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
                    color_map=k3d.matplotlib_color_maps.twilight)

    pytest.plot += mesh

    compare('mesh_attribute_advanced')

    pytest.plot.clipping_planes = [
        [1, 1, 0, 0]
    ]

    compare('mesh_attribute_advanced_clipping_planes')


def test_mesh_triangle_attribute():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    qualityFilter = vtk.vtkMeshQuality()
    qualityFilter.SetInputData(reader.GetOutput())
    qualityFilter.SetTriangleQualityMeasureToArea()
    qualityFilter.SetQuadQualityMeasureToArea()
    qualityFilter.Update()

    mesh = k3d.vtk_poly_data(qualityFilter.GetOutput(), cell_color_attribute=('Quality', 0.0, 0.83),
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += mesh

    compare('mesh_triangle_attribute')


def test_mesh_volume_data():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('./test/assets/volume.vti')
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = numpy_support.vtk_to_numpy(
        vti.GetPointData().GetArray(0)
    ).reshape(-1, y, x).astype(np.float32)

    mesh = k3d.vtk_poly_data(poly, color=0xffffff, volume=volume_data,
                             transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
                             volume_bounds=[-50, 150, -200, 100, -50, 250])

    pytest.plot += mesh

    compare('mesh_volume_data')
