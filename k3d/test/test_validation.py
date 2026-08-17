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
