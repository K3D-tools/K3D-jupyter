import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

CAMERA = [3.0, -4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_snapshot_roundtrip():
    """Render, snapshot, wipe, load - the render afterwards must still match.

    Both renders are compared against the same reference, so if the second one still matches
    then the snapshot carried everything that affects the image. camera_factor=None keeps
    compare() from resetting the camera, since the explicit camera is part of what the
    snapshot is supposed to preserve.
    """
    prepare()

    pytest.plot += k3d.points(
        np.array([[0, 0, 0], [1, 1, 1], [-1, 0.5, 0.5]], dtype=np.float32),
        point_size=0.3,
        color=0x3F6BFA,
    )
    pytest.plot += k3d.line(
        np.array([[-1, -1, 0], [0, 1, 0.5], [1, -1, 1]], dtype=np.float32),
        width=0.05,
        color=0xE6006E,
    )
    pytest.plot += k3d.mesh(
        np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.uint32),
        color=0x00A86B,
    )

    pytest.plot.background_color = 0xEEDDCC
    pytest.plot.grid_visible = False
    pytest.plot.camera_fov = 55.0
    pytest.plot.camera = list(CAMERA)

    compare("snapshot_roundtrip", camera_factor=None)

    data = pytest.plot.get_binary_snapshot()

    while len(pytest.plot.objects) > 0:
        pytest.plot -= pytest.plot.objects[-1]
    pytest.plot.background_color = 0xFFFFFF
    pytest.plot.grid_visible = True
    pytest.plot.camera_fov = 60.0
    pytest.plot.camera = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    pytest.plot.load_binary_snapshot(data)

    compare("snapshot_roundtrip", camera_factor=None)
