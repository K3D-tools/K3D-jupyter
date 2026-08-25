"""What comes back out of k3d has to be usable, whatever went in.

Two properties, both of which used to hold only by accident:

Writability. np.frombuffer inherits the mutability of the buffer it is handed, and
zlib.decompress returns immutable bytes - so whether the caller could edit an array it got back
depended on the compression level the array had been written with. Uncompressed gave a writable
array, compressed raised "assignment destination is read-only" on the first in-place edit.

Layout. Nothing promises a user's array is C-contiguous: a Fortran-ordered volume, a transposed
view or a strided slice are all perfectly sane inputs, and the values have to survive them.
"""
import numpy as np
import pytest

import k3d
from k3d.helpers import array_to_json, json_to_array

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
