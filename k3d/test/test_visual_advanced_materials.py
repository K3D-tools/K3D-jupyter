"""The roughness x metalness response under the advanced renderer."""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare


def test_material_matrix():
    prepare()
    pytest.plot.renderer = "advanced"
    pytest.plot.environment = "studio"

    for row, metalness in enumerate((0.0, 1.0)):
        for col, roughness in enumerate((0.05, 0.3, 0.7)):
            sphere = k3d.points(
                np.array([[col * 1.2, 0, row * 1.2]], dtype=np.float32),
                point_size=1.0,
                shader="mesh",
                mesh_detail=3,
                color=0xB08040,
                roughness=roughness,
                metalness=metalness,
            )
            pytest.plot += sphere

    compare("advanced_material_matrix", modes=("advanced",))
