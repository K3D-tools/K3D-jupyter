"""Resource accounting for repeated voxel updates.

Voxels.update() resolves opacity only, so assigning `voxels` rebuilds the whole object. The
meshes live in a group, and disposing only the top level released nothing, so every assignment
leaked a geometry and a colormap texture until the browser ran out of GPU memory.
"""

import numpy as np
import pytest

from .plot_compare import prepare

import k3d

SHAPE = (8, 8, 8)


def _pattern(step):
    voxels = np.zeros(SHAPE, dtype=np.uint8)
    voxels[step % SHAPE[0]] = 1

    return voxels


def _renderer_memory():
    return pytest.headless.browser.execute_script(
        "const memory = K3DInstance.getWorld().renderer.info.memory;"
        "return [memory.geometries, memory.textures];"
    )


def test_voxels_update_does_not_leak_gpu_resources():
    prepare()

    voxels = k3d.voxels(_pattern(0), color_map=[0xFF0000])
    pytest.plot += voxels
    pytest.headless.sync(hold_until_refreshed=True)

    geometries, textures = _renderer_memory()

    for step in range(1, 6):
        voxels.voxels = _pattern(step)
        pytest.headless.sync(hold_until_refreshed=True)

    after_geometries, after_textures = _renderer_memory()

    assert after_geometries <= geometries, (
        "%d geometries after 5 voxel updates, started from %d"
        % (after_geometries, geometries)
    )
    assert after_textures <= textures, (
        "%d textures after 5 voxel updates, started from %d" % (after_textures, textures)
    )
