import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

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
            np.arange(N * N - N - 1) + 0,
            np.arange(N * N - N - 1) + N,
            np.arange(N * N - N - 1) + N + 1,
            np.arange(N * N - N - 1) + 0,
            np.arange(N * N - N - 1) + N + 1,
            np.arange(N * N - N - 1) + 1,
        ]
    ).T
).astype(np.uint32)

colors = np.linspace(0, 0xFFFFFF, N * N).astype(np.uint32)


def test_lines_simple():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices, indices, flat_shading=False, shader="simple", color=0xFF
    )
    pytest.plot += lines

    compare("lines_simple", camera_factor=0.5)


def test_lines_simple_colors():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices, indices, flat_shading=False, shader="simple", colors=colors
    )
    pytest.plot += lines

    compare("lines_simple_colors", camera_factor=0.5)


def test_lines_simple_attribute():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="simple",
        attribute=phi,
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_simple_attribute", camera_factor=0.5)


def test_lines_simple_attribute_segment():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="simple",
        attribute=phi,
        indices_type="segment",
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_simple_attribute_segment", camera_factor=0.5)


def test_lines_thick():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices, indices, flat_shading=False, shader="thick", width=0.003, color=0xFF
    )
    pytest.plot += lines

    compare("lines_thick", camera_factor=0.5)


def test_lines_thick_colors():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="thick",
        width=0.003,
        colors=colors,
    )
    pytest.plot += lines

    compare("lines_thick_colors", camera_factor=0.5)


def test_lines_thick_attribute():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="thick",
        width=0.003,
        attribute=phi,
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_thick_attribute", camera_factor=0.5)


def test_lines_thick_attribute_segment():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="thick",
        width=0.003,
        attribute=phi,
        indices_type="segment",
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_thick_attribute_segment", camera_factor=0.5)


def test_lines_mesh():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices, indices, flat_shading=False, shader="mesh", width=0.003, color=0xFF
    )
    pytest.plot += lines

    compare("lines_mesh", camera_factor=0.5)

    lines.shininess = 1500.0
    lines.width = 0.02

    compare("lines_mesh_dynamic_shininess", camera_factor=0.5)


def test_lines_mesh_colors():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="mesh",
        width=0.003,
        color=0xFF,
        colors=colors,
    )
    pytest.plot += lines

    compare("lines_mesh_colors", camera_factor=0.5)

    lines.shininess = 1500.0
    lines.width = 0.02

    compare("lines_mesh_colors_dynamic_shininess", camera_factor=0.5)


def test_lines_mesh_attribute():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="mesh",
        width=0.003,
        color=0xFF,
        attribute=phi,
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_mesh_attribute", camera_factor=0.5)


def test_lines_mesh_attribute_segment():
    global vertices, indices

    prepare()

    lines = k3d.lines(
        vertices,
        indices,
        flat_shading=False,
        shader="mesh",
        width=0.003,
        color=0xFF,
        attribute=phi,
        indices_type="segment",
        color_map=k3d.matplotlib_color_maps.twilight,
    )
    pytest.plot += lines

    compare("lines_mesh_attribute_segment", camera_factor=0.5)
