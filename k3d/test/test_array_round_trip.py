"""What comes back out of k3d has to be usable, whatever went in.

Two properties, both of which used to hold only by accident:

Writability. np.frombuffer inherits the mutability of the buffer it is handed, and
zlib.decompress returns immutable bytes - so whether the caller could edit an array it got back
depended on the compression level the array had been written with. Uncompressed gave a writable
array, compressed raised "assignment destination is read-only" on the first in-place edit.

Layout. Nothing promises a user's array is C-contiguous: a Fortran-ordered volume, a transposed
view or a strided slice are all perfectly sane inputs, and the values have to survive them.
"""
import zlib

import numpy as np
import pytest

import k3d
from k3d.helpers import COMM_BUDGET_BYTES, array_to_json, json_to_array

BASE = (np.arange(24, dtype=np.float32).reshape(8, 3) * 0.5)


def layouts():
    return [
        ('c_contiguous', np.ascontiguousarray(BASE)),
        ('fortran', np.asfortranarray(BASE)),
        ('transposed_view', np.ascontiguousarray(BASE.T).T),
        ('strided', np.ascontiguousarray(np.repeat(BASE, 2, axis=0))[::2]),
        ('read_only', np.frombuffer(BASE.tobytes(), dtype=np.float32).reshape(BASE.shape)),
        ('big_endian', BASE.astype('>f4')),
        ('float64', BASE.astype(np.float64)),
    ]


@pytest.mark.parametrize('compression_level', [0, 1, 9])
def test_a_round_tripped_array_can_be_edited(compression_level):
    back = json_to_array(array_to_json(BASE, compression_level=compression_level))

    assert back.flags['WRITEABLE'], 'compression level decided whether the array was editable'

    back[0, 0] = 42.0

    assert back[0, 0] == 42.0


@pytest.mark.parametrize('name,array', layouts())
def test_every_sane_layout_survives_the_round_trip(name, array):
    expected = np.asarray(array, dtype=np.float32)

    for compression_level in (0, 1):
        back = json_to_array(array_to_json(array, compression_level=compression_level))

        assert back.shape == expected.shape, (name, compression_level)
        assert np.array_equal(back.astype(np.float32), expected), (name, compression_level)


def test_a_loaded_snapshot_hands_back_editable_arrays():
    plot = k3d.plot()
    plot += k3d.points(np.ascontiguousarray(BASE), point_size=0.2)

    loaded = k3d.plot()
    loaded.load_binary_snapshot(plot.get_binary_snapshot(compression_level=1))
    positions = loaded.objects[0].positions

    assert np.array_equal(positions, BASE)
    assert positions.flags['WRITEABLE']

    # the whole point: a notebook that loads a file and nudges the data must not have to copy
    positions[0, 0] = 42.0

    assert positions[0, 0] == 42.0


def test_oversized_volume_compresses_under_comm_budget():
    data = np.ones((512, 512, 128), dtype=np.float32)
    vol = k3d.volume(data)
    serialized = vol.get_binary()['volume']

    assert 'data' not in serialized
    assert len(serialized['compressed_data']) <= COMM_BUDGET_BYTES
    assert np.array_equal(json_to_array(serialized), data)
    assert vol.compression_level == 0


def test_payload_within_comm_budget_stays_uncompressed():
    exact = k3d.volume(np.ones((512, 256, 128), dtype=np.float32))
    serialized = exact.get_binary()['volume']

    assert isinstance(serialized['data'], memoryview)
    assert serialized['data'].nbytes == COMM_BUDGET_BYTES
    assert 'compressed_data' not in serialized
    assert exact.compression_level == 0

    # float64 narrows to float32 before the budget is measured
    narrowed = array_to_json(np.ones((512, 256, 128), dtype=np.float64))
    assert isinstance(narrowed['data'], memoryview)
    assert narrowed['data'].nbytes == COMM_BUDGET_BYTES
    assert 'compressed_data' not in narrowed


@pytest.mark.parametrize('compression_level', [1, 9])
def test_explicit_compression_level_is_preserved(compression_level):
    small = k3d.volume(BASE.reshape(2, 3, 4), compression_level=compression_level)
    assert small.get_binary()['volume']['compressed_data'] == zlib.compress(
        small.volume.ravel(), compression_level
    )
    assert small.compression_level == compression_level

    large = np.ones((512, 257, 128), dtype=np.float32)
    assert array_to_json(large, compression_level=compression_level)[
        'compressed_data'
    ] == zlib.compress(large.ravel(), compression_level)


def test_incompressible_payload_over_comm_budget_raises():
    raw = np.frombuffer(
        np.random.default_rng(0).bytes(COMM_BUDGET_BYTES + 4096), dtype=np.uint8
    )

    with pytest.raises(ValueError, match='float16') as exc:
        array_to_json(raw)

    assert f'{raw.nbytes} bytes raw' in str(exc.value)
    assert 'bytes compressed' in str(exc.value)

    vol_data = np.frombuffer(
        np.random.default_rng(0).bytes(512 * 257 * 128 * 4), dtype=np.float32
    ).reshape(512, 257, 128)
    with pytest.raises(ValueError, match='float16'):
        k3d.volume(vol_data, color_range=[0.0, 1.0])
