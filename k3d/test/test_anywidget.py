import numpy as np
import pytest

import k3d
from k3d._widget import K3DAnyWidget
from k3d.objects.base import VoxelChunk

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def test_esm_is_packaged():
    from k3d._widget import _STATIC

    assert (_STATIC / "widget.mjs").exists()


def test_widget_kinds():
    assert k3d.plot()._kind == "plot"
    assert k3d.mesh(VERTICES, INDICES)._kind == "object"
    assert VoxelChunk(voxels=np.ones((2, 2, 2), dtype=np.uint8),
                      coord=np.zeros(3, dtype=np.uint32), multiple=1)._kind == "chunk"
    assert k3d.transfer_function_editor()._kind == "tf_editor"


def test_synced_props_enumerate_public_traits():
    mesh = k3d.mesh(VERTICES, INDICES)

    assert "vertices" in mesh._synced_props
    assert "indices" in mesh._synced_props
    assert "compression_level" in mesh._synced_props
    assert not any(name.startswith("_") for name in mesh._synced_props)


def test_binary_snapshot_carries_no_transport_keys():
    mesh = k3d.mesh(VERTICES, INDICES)

    assert not any(key.startswith("_") for key in mesh.get_binary())


def test_snapshot_roundtrip_strips_transport_keys():
    plot = k3d.plot()
    plot += k3d.mesh(VERTICES, INDICES)

    snapshot = plot.get_binary_snapshot(1)

    plot2 = k3d.plot()
    plot2.load_binary_snapshot(snapshot)

    assert len(plot2.objects) == 1
    assert np.allclose(plot2.objects[0].vertices, VERTICES)
