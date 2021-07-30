import k3d
import numpy as np
import pytest
from .plot_compare import *
import math

a = 1.5
b = -1.8
c = 1.6
d = 0.9

N = int(1e4)

positions = np.zeros(N * 3, dtype=np.float32)
opacities = np.zeros(N, dtype=np.float32)

x = 0
y = 0
z = 0

for i in range(N):
    xn = math.sin(a * y) + c * math.cos(a * x)
    yn = math.sin(b * x) + d * math.cos(b * y)
    zn = math.sin(1.4 * (x + z))

    positions[i * 3] = xn
    positions[i * 3 + 1] = yn
    positions[i * 3 + 2] = zn
    opacities[i] = math.sqrt(xn ** 2 + yn ** 2 + zn ** 2) / 2.0

    x = xn
    y = yn
    z = zn


def test_points_flat():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='flat', point_size=0.05, opacities=opacities,
                        color=0xff0000)

    pytest.plot += points

    compare('points_flat')


def test_points_3d():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='3d', point_size=0.1, opacities=opacities,
                        color=0xff0000)

    pytest.plot += points

    compare('points_3d')


def test_points_3d_clipping_plane():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='3d', point_size=0.1, opacities=opacities,
                        color=0xff0000)
    pytest.plot.clipping_planes = [
        [1, 1, 0, 0]
    ]
    pytest.plot += points

    compare('points_3d_clipping_plane')


def test_points_3dSpecular():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='3dSpecular', point_size=0.1, opacities=opacities,
                        color=0xff0000)

    pytest.plot += points

    compare('points_3dSpecular')


def test_points_3dSpecular_sizes():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='3dSpecular', opacities=opacities,
                        point_sizes=np.linspace(0, 0.1, N), color=0xff0000)

    pytest.plot += points

    compare('points_3dSpecular_sizes')


def test_points_mesh_sizes():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='mesh', opacities=opacities,
                        point_sizes=np.linspace(0, 0.1, N), color=0xff0000)

    pytest.plot += points

    compare('points_mesh_sizes')


def test_points_mesh():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='mesh', point_size=0.1, opacities=opacities,
                        color=0xff0000)

    pytest.plot += points

    compare('points_mesh')


def test_points_mesh_clipping_plane():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='mesh', point_size=0.1, opacities=opacities,
                        color=0xff0000)
    pytest.plot.clipping_planes = [
        [1, 1, 0, 0]
    ]
    pytest.plot += points

    compare('points_mesh_clipping_plane')


def test_points_mesh_low_detail():
    global positions, opacities

    prepare()

    points = k3d.points(np.array([[0, 0, 0], [1, 0, 0]]), shader='mesh', point_size=0.3,
                        opacities=opacities,
                        mesh_detail=1, color=0xff0000)

    pytest.plot += points

    compare('points_mesh_low_detail')


def test_points_mesh_high_detail():
    global positions, opacities

    prepare()

    points = k3d.points(np.array([[0, 0, 0], [1, 0, 0]]), shader='mesh', point_size=0.3,
                        opacities=opacities,
                        mesh_detail=8, color=0xff0000)

    pytest.plot += points

    compare('points_mesh_high_detail')


def test_points_dot():
    global positions, opacities

    prepare()

    points = k3d.points(positions, shader='dot', point_size=3, opacities=opacities,
                        color=0xff0000)

    pytest.plot += points

    compare('points_dot')
