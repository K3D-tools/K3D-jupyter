import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

vertices = [
    -10.0,
    -15.0,
    0.0,
    -10.0,
    5.0,
    0.0,
    10.0,
    5.0,
    0.0,
    -10.0,
    -15.0,
    0.0,
    10.0,
    -15.0,
    0.0,
    -10.0,
    5.0,
    0.0,
    0.0,
    15.0,
    0.0,
    10.0,
    5.0,
    0.0,
    10.0,
    -15.0,
    0.0,
]

colors = [
    0xFF,
    0xFFFF,
    0xFF00FF,
    0x00FFFF,
    0xFFFF00,
    0xFF00,
    0xFF0000,
    0xFF00FF,
    0xFFFFFF,
]


def test_line_simple_red():
    global vertices

    prepare()

    lines = k3d.line(
        vertices,
        color=0xFF0000,
        shader="simple",
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_simple_red")


def test_line_simple():
    global vertices, colors

    prepare()

    lines = k3d.line(
        vertices,
        colors=colors,
        shader="simple",
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_simple")


def test_line_simple_clipping_plane():
    global vertices, colors

    prepare()
    pytest.plot.clipping_planes = [[1, 1, 0, 0]]

    lines = k3d.line(
        vertices,
        colors=colors,
        shader="simple",
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_simple_clipping_plane")


def test_line_thick():
    global vertices, colors

    prepare()

    lines = k3d.line(
        vertices,
        colors=colors,
        width=0.5,
        shader="thick",
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_thick")


def test_line_thick_clipping_plane():
    global vertices, colors

    prepare()
    pytest.plot.clipping_planes = [[1, 1, 0, 0]]
    lines = k3d.line(
        vertices,
        colors=colors,
        width=0.5,
        shader="thick",
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_thick_clipping_plane")


def test_line_mesh():
    global vertices, colors

    prepare()

    lines = k3d.line(
        vertices,
        colors=colors,
        width=0.5,
        shader="mesh",
        radial_segments=8,
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_mesh")

    lines.shininess = 500.0

    compare("line_mesh_dynamic_shininess")


def test_line_mesh_clipping_plane():
    global vertices, colors

    prepare()
    pytest.plot.clipping_planes = [[1, 1, 0, 0]]
    lines = k3d.line(
        vertices,
        colors=colors,
        width=0.5,
        shader="mesh",
        radial_segments=8,
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_mesh_clipping_plane")


def test_line_simplified_mesh():
    global vertices, colors

    prepare()

    lines = k3d.line(
        vertices,
        colors=colors,
        width=0.5,
        shader="mesh",
        radial_segments=3,
        transform=k3d.transform(rotation=(np.pi / 2, 1, 0, 0)),
    )
    pytest.plot += lines

    compare("line_simplified_mesh")


def test_line_advanced():
    import math

    prepare()

    for i in range(10):
        v = []
        s = []

        for fi in np.arange(0, 2 * math.pi, 0.01):
            v.append(
                [
                    math.sin(fi * 4 + i),
                    math.cos(fi * 7 + i),
                    math.cos(fi * 3) + fi * 1.5,
                ]
            )
            s.append(math.sin(fi * i))

        pytest.plot += k3d.line(
            v,
            attribute=s,
            color_range=[-1, 1],
            color_map=k3d.basic_color_maps.Jet,
            width=0.03,
        )

    compare("line_advanced")


def test_line_mesh_opacity_no_depth_peels():
    import math

    prepare()

    for i in range(10):
        v = []
        s = []

        for fi in np.arange(0, 2 * math.pi, 0.01):
            v.append(
                [
                    math.sin(fi * 4 + i),
                    math.cos(fi * 7 + i),
                    math.cos(fi * 3) + fi * 1.5,
                ]
            )
            s.append(math.sin(fi * i))

        pytest.plot += k3d.line(
            v,
            shader="mesh",
            opacity=0.25,
            attribute=s,
            color_range=[-1, 1],
            color_map=k3d.basic_color_maps.Jet,
            width=0.06,
        )

    pytest.plot.camera = [-0.052159935763831905,
                          3.1673584843034166,
                          5.356722426260206,
                          -2.8759241104125977e-05,
                          1.1920928955078125e-07,
                          5.431476727128029,
                          -0.996488124472286,
                          0.037236633436333995,
                          -0.07499900610035434]

    compare("line_mesh_opacity_no_depth_peels", camera_factor=0.35)


def test_line_mesh_opacity_depth_peels():
    import math

    prepare(depth_peels=8)

    for i in range(10):
        v = []
        s = []

        for fi in np.arange(0, 2 * math.pi, 0.01):
            v.append(
                [
                    math.sin(fi * 4 + i),
                    math.cos(fi * 7 + i),
                    math.cos(fi * 3) + fi * 1.5,
                ]
            )
            s.append(math.sin(fi * i))

        pytest.plot += k3d.line(
            v,
            shader="mesh",
            opacity=0.25,
            attribute=s,
            color_range=[-1, 1],
            color_map=k3d.basic_color_maps.Jet,
            width=0.06,
        )

    pytest.plot.camera = [-0.052159935763831905,
                          3.1673584843034166,
                          5.356722426260206,
                          -2.8759241104125977e-05,
                          1.1920928955078125e-07,
                          5.431476727128029,
                          -0.996488124472286,
                          0.037236633436333995,
                          -0.07499900610035434]

    compare("line_mesh_opacity_depth_peels", camera_factor=0.35)
