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


def test_big_endian_arrays_convert_quietly():
    """Big-endian arrays (legacy VTK) convert to the native dtype without a spurious
    "does not match" warning: traittypes names dtypes without their byte order."""
    import warnings

    vertices = np.zeros((3, 3), dtype=">f4")
    assert vertices.dtype != np.dtype(np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = k3d.mesh(vertices, np.array([[0, 1, 2]], dtype=np.uint32))

    assert mesh.vertices.dtype == np.dtype(np.float32)
    assert not [w for w in caught if "does not match" in str(w.message)]


def test_numpy_integers_are_accepted_as_ints():
    obj = k3d.volume(DATA, compression_level=np.int32(1))

    assert obj.compression_level == 1


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint16, np.float32])
def test_voxels_above_uint8_range_are_rejected_not_wrapped(dtype):
    data = np.ones((2, 2, 2), dtype=dtype)
    data[1, 1, 1] = 256

    with pytest.raises(TraitError):
        k3d.voxels(data)

    obj = k3d.voxels(np.ones((2, 2, 2), dtype=np.uint8))

    with pytest.raises(TraitError):
        obj.voxels = data


def test_voxels_at_uint8_limit_are_kept():
    data = np.zeros((2, 2, 2), dtype=np.int64)
    data[0, 0, 0] = 255

    assert k3d.voxels(data).voxels[0, 0, 0] == 255


def test_colors_above_uint32_range_are_rejected_not_wrapped():
    positions = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(TraitError):
        k3d.points(positions, colors=np.array([2**32 + 0xFF, 0], dtype=np.int64))


def test_voxel_chunk_above_uint8_range_is_rejected_not_wrapped():
    with pytest.raises(TraitError):
        k3d.voxel_chunk(np.array([[[1, 256]]], dtype=np.int64), [0, 0, 0])


def test_voxel_chunk_still_takes_lists():
    chunk = k3d.voxel_chunk([[[1, 255]]], [0, 0, 0])

    assert chunk.voxels.dtype == np.uint8
    assert chunk.voxels.ravel().tolist() == [1, 255]


def test_voxel_chunk_conversion_does_not_warn(recwarn):
    k3d.voxel_chunk(np.array([[[1, 255]]], dtype=np.int64), [0, 0, 0])
    k3d.voxel_chunk([[[1, 255]]], [0, 0, 0])

    assert len(recwarn) == 0


def test_voxel_chunk_negative_value_is_rejected():
    with pytest.raises(TraitError):
        k3d.voxel_chunk(np.array([[[-1, 2]]], dtype=np.int64), [0, 0, 0])
