import k3d
import numpy as np
import pytest
from .plot_compare import *
import math

v = []
s = []
o = []

for i in range(10):
    for fi in np.arange(0, 2 * math.pi, 0.1):
        v.append([
            math.sin(fi * 4 + i),
            math.cos(fi * 7 + i),
            math.cos(fi * 3) + fi * 1.5
        ])
        s.append(math.sin(fi * i))
        o.append(abs(math.cos(fi * i)))

v = np.array(v, dtype=np.float32)
s = np.array(s, dtype=np.float32)
o = np.array(o, dtype=np.float32)


def test_points_flat():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='flat', point_size=0.1, opacities=o,
                        color=0xff0000)

    pytest.plot += points

    compare('points_flat')


def test_points_3d():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3d', point_size=0.2, opacities=o,
                        color=0xff0000)

    pytest.plot += points

    compare('points_3d')


def test_points_3d_no_opacity():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3d', point_size=0.4, color=0xff0000)

    pytest.plot += points

    compare('points_3d_no_opacity')


def test_points_3d_no_opacity_with_plane():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3d', point_size=0.4, color=0xff0000)
    plane = k3d.mesh([-1, 0, -2,
                      1, 0, -2,
                      1, 0, 12,
                      -1, 0, 12
                      ],
                     [0, 1, 2,
                      2, 3, 0], color=0x00ff00)

    pytest.plot += points
    pytest.plot += plane

    compare('points_3d_no_opacity_with_plane')


def test_points_3d_clipping_plane():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3d', point_size=0.2, opacities=o,
                        color=0xff0000)
    pytest.plot.clipping_planes = [
        [1, 1, 0, 0]
    ]
    pytest.plot += points

    compare('points_3d_clipping_plane')


def test_points_3dSpecular():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3dSpecular', point_size=0.2, opacities=o,
                        color=0xff0000)

    pytest.plot += points

    compare('points_3dSpecular')


def test_points_3dSpecular_sizes():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='3dSpecular', opacities=o,
                        point_sizes=np.linspace(0, 0.2, v.shape[0]), color=0xff0000)

    pytest.plot += points

    compare('points_3dSpecular_sizes')


def test_points_mesh_sizes():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='mesh', opacities=o,
                        point_sizes=np.linspace(0, 0.2, v.shape[0]), color=0xff0000)

    pytest.plot += points

    compare('points_mesh_sizes')


def test_points_mesh():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='mesh', point_size=0.2, opacities=o,
                        color=0xff0000)

    pytest.plot += points

    compare('points_mesh')


def test_points_mesh_clipping_plane():
    global v, s, o

    prepare()

    points = k3d.points(v, shader='mesh', point_size=0.2, opacities=o,
                        color=0xff0000)
    pytest.plot.clipping_planes = [
        [1, 1, 0, 0]
    ]
    pytest.plot += points

    compare('points_mesh_clipping_plane')


def test_points_mesh_low_detail():
    prepare()

    points = k3d.points(np.array([[0, 0, 0], [1, 0, 0]]), shader='mesh', point_size=0.3,
                        mesh_detail=1, color=0xff0000)

    pytest.plot += points

    compare('points_mesh_low_detail')


def test_points_mesh_high_detail():
    prepare()

    points = k3d.points(np.array([[0, 0, 0], [1, 0, 0]]), shader='mesh', point_size=0.3,
                        mesh_detail=8, color=0xff0000)

    pytest.plot += points

    compare('points_mesh_high_detail')


def test_points_dot():
    global v, s

    prepare()

    points = k3d.points(v, shader='dot', point_size=3, opacities=o,
                        color=0xff0000)

    pytest.plot += points

    compare('points_dot')
