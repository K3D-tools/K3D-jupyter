"""The impostor sphere against a real mesh sphere: same silhouette, same diffuse shading.

They are deliberately no longer lit identically. The impostor evaluates a specular lobe for
the key light only - four lobes per fragment is a lot for a sprite a few dozen pixels across
- while a mesh goes through three's MeshStandardMaterial and keeps all four. What this
guards is the rest: that the analytic sphere and the tessellated one agree in size, position
and body shading, which is where an impostor bug would show.
"""

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
