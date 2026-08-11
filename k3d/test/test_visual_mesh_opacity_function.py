"""Visual coverage for opacity_function on a plain mesh.

The trait was synced but ignored: the colormap LUT is built without an alpha channel and the
material stays opaque. Both tests end in the same state and share one reference, so the
creation path and the update path are asserted to agree.
"""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

# Upright, facing the default camera, so the grid stays visible through the faded half.
VERTICES = np.array(
    [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
    dtype=np.float32,
)
INDICES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

# Ramps with x, so the LUT is sampled end to end across the quad.
ATTRIBUTE = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

# Fully transparent at the low end, fully opaque at the high end.
OPACITY_FUNCTION = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)


def _mesh(**kwargs):
    return k3d.mesh(
        VERTICES,
        INDICES,
        attribute=ATTRIBUTE,
        color_map=k3d.colormaps.matplotlib_color_maps.Viridis,
        color_range=[0.0, 1.0],
        flat_shading=False,
        **kwargs,
    )


def test_mesh_opacity_function():
    prepare()
    pytest.plot += _mesh(opacity_function=OPACITY_FUNCTION)

    compare("mesh_opacity_function")


def test_mesh_opacity_function_update():
    prepare()
    obj = _mesh()
    pytest.plot += obj

    compare("mesh_attribute_opaque")

    # No opacity_function at creation means an opaque material, so this has to rebuild the
    # object rather than patch the LUT in place.
    obj.opacity_function = OPACITY_FUNCTION

    compare("mesh_opacity_function")
