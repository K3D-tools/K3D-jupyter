"""Scenes only the cinematic renderer has: the volume hybrid and the path-tracing
budgets. Everything else is covered by the cinematic mode of the shared visual suite."""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare


def _sphere(radius=1.0, offset=(0.0, 0.0, 0.0), rings=24, segments=48):
    phi = np.linspace(0, np.pi, rings, dtype=np.float32)
    theta = np.linspace(0, 2 * np.pi, segments, dtype=np.float32)
    P, T = np.meshgrid(phi, theta)
    vertices = np.stack(
        [np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], axis=-1
    ).reshape(-1, 3).astype(np.float32) * radius + np.array(offset, np.float32)

    indices = []
    for i in range(segments - 1):
        for j in range(rings - 1):
            a = i * rings + j
            indices.append([a, a + rings, a + rings + 1])
            indices.append([a, a + rings + 1, a + 1])

    return vertices, np.array(indices, dtype=np.uint32)


def _cloud(size=40, falloff=3.0):
    gx, gy, gz = np.mgrid[-1:1:size * 1j, -1:1:size * 1j, -1:1:size * 1j]
    return np.exp(-falloff * (gx ** 2 + gy ** 2 + gz ** 2)).astype(np.float32)


def test_cinematic_volume_hybrid():
    """Volumes are not path traced: the march stops at the first traced hit and composites
    over the accumulation. The sphere overlaps the cloud, so both failure directions show."""
    prepare()

    vertices, indices = _sphere(0.5, (0.7, 0.0, 0.0))

    pytest.plot += k3d.volume(
        _cloud(), color_range=[0.05, 1.0], color_map=k3d.basic_color_maps.Jet,
        alpha_coef=20.0, bounds=[-1, 1, -1, 1, -1, 1],
    )
    pytest.plot += k3d.mesh(vertices, indices, color=0xC28439, roughness=0.35)

    compare("cinematic_volume_hybrid", modes=("cinematic",))


def test_cinematic_mip_hybrid():
    prepare()

    vertices, indices = _sphere(0.5, (0.7, 0.0, 0.0))

    pytest.plot += k3d.mip(
        _cloud(), color_range=[0.05, 1.0], color_map=k3d.basic_color_maps.CoolWarm,
        bounds=[-1, 1, -1, 1, -1, 1],
    )
    pytest.plot += k3d.mesh(vertices, indices, color=0xC28439, roughness=0.35)

    compare("cinematic_mip_hybrid", modes=("cinematic",))


def test_cinematic_single_bounce():
    """One bounce is direct lighting only: no colour bleeding between the walls and sphere."""
    prepare()

    vertices, indices = _sphere(0.6)
    floor = np.array([[-2, -2, -0.6], [2, -2, -0.6], [2, 2, -0.6], [-2, 2, -0.6]],
                     np.float32)
    wall = np.array([[-2, 2, -0.6], [2, 2, -0.6], [2, 2, 2], [-2, 2, 2]], np.float32)
    quad = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)

    pytest.plot += k3d.mesh(vertices, indices, color=0xEEEEEE, roughness=0.4)
    pytest.plot += k3d.mesh(floor, quad, color=0xCC2200, roughness=0.9)
    pytest.plot += k3d.mesh(wall, quad, color=0x2244CC, roughness=0.9)
    pytest.plot.cinematic_bounces = 1

    compare("cinematic_single_bounce", modes=("cinematic",))


def test_cinematic_mixed_vertex_attributes():
    """The tracer merges the whole proxy scene and takes its attribute set from the first
    geometry, so every object must be given the same one: a flat tube (colours, no uv), a
    colormapped tube (both) and icospheres (uv, no colours) have to keep their own colours
    whatever order they arrive in."""
    prepare()

    path = np.array([[-1, -1, 0], [0, 1, 0.5], [1, -1, 1]], dtype=np.float32)

    pytest.plot += k3d.line(path, width=0.05, color=0xE6006E)
    pytest.plot += k3d.line(
        path + np.array([0, 0, -0.8], dtype=np.float32), width=0.05,
        attribute=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        color_range=[0.0, 1.0], color_map=k3d.basic_color_maps.Jet,
    )
    pytest.plot += k3d.points(path, point_size=0.25, color=0x3F6BFA)

    compare("cinematic_mixed_vertex_attributes", modes=("cinematic",))
