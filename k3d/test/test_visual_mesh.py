import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

import k3d
from k3d.helpers import download
from .plot_compare import compare, prepare

vertices = [
    -10,
    0,
    -1,
    10,
    0,
    -1,
    10,
    0,
    1,
    -10,
    0,
    1,
]

indices = [0, 1, 3, 1, 2, 3]


def test_mesh():
    global vertices, indices

    prepare()

    mesh = k3d.mesh(vertices, indices)
    pytest.plot += mesh

    compare("mesh")


def test_mesh_attribute():
    global vertices, indices

    prepare()

    vertex_attribute = [0, 1, 1, 0]
    mesh = k3d.mesh(
        vertices,
        indices,
        attribute=vertex_attribute,
        color_map=k3d.basic_color_maps.CoolWarm,
        color_range=[0.0, 1.0],
    )
    pytest.plot += mesh

    compare("mesh_attribute")


def test_mesh_advanced():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x40E0D0,
        flat_shading=True,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    pytest.plot += mesh

    compare("mesh_advanced")


def test_mesh_advanced_smoothed():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x40E0D0,
        flat_shading=False,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    pytest.plot += mesh

    compare("mesh_advanced_smoothed")


def test_mesh_advanced_roughness():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x40E0D0,
        flat_shading=False,
        roughness=0.4,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    pytest.plot += mesh

    compare("mesh_advanced_roughness")

    mesh.roughness = 0.06

    compare("mesh_advanced_dynamic_roughness")


def test_mesh_advanced_opacity():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x40E0D0,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    pytest.plot += mesh

    compare("mesh_advanced_opacity")


def test_mesh_advanced_opacity_depth_peels():
    prepare(depth_peels=8)

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh1 = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x00ff00,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    mesh2 = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0xff0000,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0], translation=[25, 0, 0]),
    )
    pytest.plot += mesh1
    pytest.plot += mesh2

    compare("test_mesh_advanced_opacity_depth_peels")


def test_mesh_advanced_wireframe():
    prepare(depth_peels=0)

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = k3d.vtk_poly_data(
        reader.GetOutput(),
        color=0x40E0D0,
        opacity=0.2,
        wireframe=True,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )
    pytest.plot += mesh

    compare("mesh_advanced_wireframe")


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
    indices = (
            np.stack(
                [
                    np.arange(N * N) + 0,
                    np.arange(N * N) + N,
                    np.arange(N * N) + N + 1,
                    np.arange(N * N) + 0,
                    np.arange(N * N) + N + 1,
                    np.arange(N * N) + 1,
                ]
            ).T
            % (N * N)
    ).astype(np.uint32)

    mesh = k3d.mesh(
        vertices,
        indices,
        flat_shading=False,
        attribute=phi,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        color_map=k3d.matplotlib_color_maps.twilight,
    )

    pytest.plot += mesh

    compare("mesh_attribute_advanced")

    pytest.plot.clipping_planes = [[1, 1, 0, 0]]

    compare("mesh_attribute_advanced_clipping_planes")


def test_mesh_triangle_attribute():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    qualityFilter = vtk.vtkMeshQuality()
    qualityFilter.SetInputData(reader.GetOutput())
    qualityFilter.SetTriangleQualityMeasureToArea()
    qualityFilter.SetQuadQualityMeasureToArea()
    qualityFilter.Update()

    mesh = k3d.vtk_poly_data(
        qualityFilter.GetOutput(),
        cell_color_attribute=("Quality", 0.0, 0.83),
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
    )

    pytest.plot += mesh

    compare("mesh_triangle_attribute")


def test_mesh_volume_data():
    prepare()

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

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

    mesh = k3d.vtk_poly_data(
        poly,
        color=0xFFFFFF,
        volume=volume_data,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        volume_bounds=[-50, 150, -200, 100, -50, 250],
    )

    pytest.plot += mesh

    compare("mesh_volume_data")


def test_mesh_volume_data_no_depth_peels():
    prepare(depth_peels=0)

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

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

    mesh1 = k3d.vtk_poly_data(
        poly,
        volume=volume_data,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        volume_bounds=[-50, 150, -200, 100, -50, 250],
    )
    mesh2 = k3d.vtk_poly_data(
        poly,
        volume=volume_data,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0], translation=[25, 0, 0]),
        volume_bounds=[-50, 150, -200, 100, -50, 250],
    )

    pytest.plot += mesh1
    pytest.plot += mesh2

    compare("mesh_volume_data_no_depth_peels")


def test_mesh_volume_data_depth_peels():
    prepare(depth_peels=8)

    filename = download(
        "https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl"
    )

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

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

    mesh1 = k3d.vtk_poly_data(
        poly,
        volume=volume_data,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        volume_bounds=[-50, 150, -200, 100, -50, 250],
    )
    mesh2 = k3d.vtk_poly_data(
        poly,
        volume=volume_data,
        flat_shading=False,
        opacity=0.5,
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0], translation=[25, 0, 0]),
        volume_bounds=[-50, 150, -200, 100, -50, 250],
    )

    pytest.plot += mesh1
    pytest.plot += mesh2

    compare("mesh_volume_data_depth_peels")


TETRA_VERTICES = np.array(
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32
)
TETRA_INDICES = np.array(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32
)

CUBE_VERTICES = np.array(
    [
        [0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0],
        [0, 0, 3], [3, 0, 3], [3, 3, 3], [0, 3, 3],
    ],
    dtype=np.float32,
)
CUBE_INDICES = np.array(
    [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ],
    dtype=np.uint32,
)


def test_mesh_vertices_morph():
    """A vertices update of the same count is an in-place morph: the position
    buffer mutates, the scene object survives (no delete/create round trip)."""
    prepare()

    mesh = k3d.mesh(TETRA_VERTICES, TETRA_INDICES, color=0x2244AA)
    pytest.plot += mesh
    pytest.headless.sync(hold_until_refreshed=True)

    uuid_before = pytest.headless.browser.execute_script(
        "return K3DInstance.getWorld().ObjectsById[%d].uuid;" % mesh.id
    )

    mesh.vertices = TETRA_VERTICES * np.array([2.0, 0.5, 1.0], dtype=np.float32)

    compare("mesh_vertices_morph", modes=("simple",))

    uuid_after = pytest.headless.browser.execute_script(
        "return K3DInstance.getWorld().ObjectsById[%d].uuid;" % mesh.id
    )
    assert uuid_before == uuid_after


def test_mesh_vertices_indices_sequential_update():
    """vertices+indices land as two sequential updates, never one transaction.

    The inconsistent middle state (indices reaching beyond the vertex pool) must
    not hang the scene and must not throw - the mesh draws nothing until the pair
    is consistent again.
    """
    prepare()

    mesh = k3d.mesh(TETRA_VERTICES, TETRA_INDICES, color=0x2244AA)
    pytest.plot += mesh
    pytest.headless.sync(hold_until_refreshed=True)

    # the worse order on purpose: indices first, beyond the current 4 vertices
    mesh.indices = CUBE_INDICES
    compare("mesh_inconsistent_indices", modes=("simple",))

    mesh.vertices = CUBE_VERTICES
    compare("mesh_vertices_indices_updated", modes=("simple",))

    logs = pytest.headless.browser.get_log("browser")
    uncaught = [e["message"] for e in logs if "Uncaught" in e["message"]]
    assert uncaught == [], uncaught
    assert any("reaches beyond" in e["message"] for e in logs)


def test_mesh_vertices_indices_grouped_update():
    """Both arrays changed within one sync travel as a single diff and reload
    once, with no inconsistent middle state (the jupyter equivalent is
    `with mesh.hold_sync():`). Shrinking in this order would be invalid
    sequentially - grouped it must pass without the guard firing.
    """
    prepare()

    mesh = k3d.mesh(CUBE_VERTICES, CUBE_INDICES, color=0x2244AA)
    pytest.plot += mesh
    pytest.headless.sync(hold_until_refreshed=True)

    mesh.vertices = TETRA_VERTICES
    mesh.indices = TETRA_INDICES

    compare("mesh_vertices_indices_grouped", modes=("simple",))

    logs = pytest.headless.browser.get_log("browser")
    assert not any("reaches beyond" in e["message"] for e in logs)
