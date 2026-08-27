"""Where the BVH is built must not change what is traced.

Scene size decides it: above the threshold the build goes to a worker, so the main thread
keeps answering on geometry large enough to otherwise hang the browser. The threshold is
moved here instead of building millions of triangles.
"""

import hashlib

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

REBUILD = """
var mode = K3DInstance.__cinematicSpike();
mode.setWorkerThreshold(arguments[0]);
mode.invalidateScene();
"""

LAST_BUILD = 'return K3DInstance.__cinematicSpike().lastBuild();'


def _sphere(radius=1.0, rings=24, segments=48):
    phi = np.linspace(0, np.pi, rings, dtype=np.float32)
    theta = np.linspace(0, 2 * np.pi, segments, dtype=np.float32)
    P, T = np.meshgrid(phi, theta)
    vertices = np.stack(
        [np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], axis=-1
    ).reshape(-1, 3).astype(np.float32) * radius

    indices = []
    for i in range(segments - 1):
        for j in range(rings - 1):
            a = i * rings + j
            indices.append([a, a + rings, a + rings + 1])
            indices.append([a, a + rings + 1, a + 1])

    return vertices, np.array(indices, dtype=np.uint32)


def _render(threshold):
    pytest.headless.browser.execute_script(REBUILD, threshold)

    png = pytest.headless.get_screenshot()

    return (hashlib.sha256(png).hexdigest(),
            pytest.headless.browser.execute_script(LAST_BUILD))


def test_worker_built_bvh_traces_the_same_image():
    prepare()

    vertices, indices = _sphere(0.6)
    floor = np.array([[-2, -2, -0.6], [2, -2, -0.6], [2, 2, -0.6], [-2, 2, -0.6]],
                     np.float32)

    pytest.plot += k3d.mesh(vertices, indices, color=0xEEEEEE, roughness=0.35)
    pytest.plot += k3d.mesh(floor, np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                            color=0xCC5522, roughness=0.9)
    pytest.plot.renderer = 'cinematic'
    pytest.headless.sync(hold_until_refreshed=True)

    try:
        on_main_thread, main_build = _render(10 ** 9)
        in_worker, worker_build = _render(0)
    finally:
        # a threshold left moved would silently decide the build path of every later test
        pytest.headless.browser.execute_script(REBUILD, None)

    assert main_build['worker'] is False, main_build
    assert worker_build['worker'] is True, worker_build
    assert main_build['triangles'] == worker_build['triangles']
    assert in_worker == on_main_thread
