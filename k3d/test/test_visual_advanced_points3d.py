"""The stage-4 checkpoint: the impostor sphere and a real mesh sphere lit the same."""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare


def test_points3d_vs_mesh_sphere():
    prepare()

    pytest.plot += k3d.points(np.array([[-1.2, 0, 0]], dtype=np.float32),
                              point_size=2.0, shader="3d",
                              color=0xCC6633, roughness=0.3)
    pytest.plot += k3d.points(np.array([[1.2, 0, 0]], dtype=np.float32),
                              point_size=2.0, shader="mesh", mesh_detail=4,
                              color=0xCC6633, roughness=0.3)

    compare("points3d_vs_mesh_sphere")
