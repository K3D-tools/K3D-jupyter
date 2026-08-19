import numpy as np
import pytest
from traitlets import TraitError

import k3d

DATA = np.zeros((4, 4, 4), dtype=np.float32)


@pytest.mark.parametrize("factory", [k3d.volume, k3d.mip])
@pytest.mark.parametrize(
    "kwargs", [{"samples": 0}, {"samples": -1}, {"gradient_step": 0.0}]
)
def test_volumetric_sampling_must_be_positive(factory, kwargs):
    with pytest.raises(TraitError):
        factory(DATA, **kwargs)


def test_volumetric_sampling_assignment_must_be_positive():
    obj = k3d.volume(DATA)

    with pytest.raises(TraitError):
        obj.samples = 0

    with pytest.raises(TraitError):
        obj.gradient_step = -0.1


@pytest.mark.parametrize("value", [np.float32(2.5), np.float64(2.5), np.int32(2)])
def test_numpy_scalars_are_accepted_as_floats(value):
    """Float traits accept any numpy scalar, not only np.float64 (which subclasses float)."""
    positions = np.zeros((3, 3), dtype=np.float32)

    assert k3d.points(positions, point_size=value).point_size == float(value)

    obj = k3d.points(positions)
    obj.point_size = value
    assert obj.point_size == float(value)


def test_numpy_scalars_are_accepted_as_keyframes():
    obj = k3d.points(np.zeros((3, 3), dtype=np.float32))
    obj.point_size = {"0": np.float32(1.0), "1": np.float32(4.0)}

    assert obj.point_size == {"0": 1.0, "1": 4.0}


def test_numpy_integers_are_accepted_as_ints():
    obj = k3d.volume(DATA, compression_level=np.int32(1))

    assert obj.compression_level == 1
