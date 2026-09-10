"""What the headless sync decides to resend, kernel side, without a browser.

Decisions, not mechanism: an untouched frame sends nothing, and every way of editing an
array is still noticed by a diff that no longer keeps a copy of one to compare against.
"""

import gc
import weakref

import numpy as np

import k3d
from k3d.headless import _ArrayFingerprint, _snapshot, _SyncState

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def _synced(*objects):
    """A plot and its sync state, with the initial full send already done."""
    plot = k3d.plot(camera_auto_fit=False)

    for o in objects:
        plot += o

    state = _SyncState(plot)
    state.diff()

    return plot, state


def _touched(state):
    """Which properties of which object the next diff would resend."""
    return {
        object_id: sorted(set(props) - {"id", "type"})
        for object_id, props in state.diff()["objects_diff"].items()
    }


def test_first_diff_sends_the_whole_object():
    plot = k3d.plot(camera_auto_fit=False)
    mesh = k3d.mesh(VERTICES, INDICES)
    plot += mesh

    sent = _SyncState(plot).diff()["objects_diff"][mesh.id]

    assert set(sent) == set(mesh._synced_props)


def test_an_untouched_frame_sends_nothing():
    volume = k3d.volume(np.zeros((16, 16, 16), np.float32), color_range=[0, 1])
    _, state = _synced(volume, k3d.mesh(VERTICES, INDICES))

    assert _touched(state) == {}
    assert _touched(state) == {}  # and stays that way, sync after sync


def test_reassigning_the_same_values_is_not_a_change():
    mesh = k3d.mesh(VERTICES, INDICES)
    _, state = _synced(mesh)

    mesh.vertices = np.array(mesh.vertices)

    assert _touched(state) == {}


def test_a_reassigned_array_is_sent():
    mesh = k3d.mesh(VERTICES, INDICES)
    _, state = _synced(mesh)

    mesh.vertices = VERTICES * 2

    assert _touched(state) == {mesh.id: ["vertices"]}


def test_an_in_place_edit_is_sent():
    volume = k3d.volume(np.zeros((16, 16, 16), np.float32), color_range=[0, 1])
    _, state = _synced(volume)

    volume.volume[0, 0, 0] = 1.0

    assert _touched(state) == {volume.id: ["volume"]}


def test_an_in_place_edit_that_writes_the_same_value_is_not_sent():
    volume = k3d.volume(np.zeros((16, 16, 16), np.float32), color_range=[0, 1])
    _, state = _synced(volume)

    volume.volume[0, 0, 0] = 0.0

    assert _touched(state) == {}


def test_an_in_place_edit_of_a_large_array_is_sent():
    # past 8 MB a checksum lane covers more than one word - the case a lane sum could hide
    data = np.zeros((256, 256, 256), np.float32)  # 64 MB
    volume = k3d.volume(data, color_range=[0, 1])
    _, state = _synced(volume)

    data[128, 128, 128] = np.float32(1e-30)

    assert _touched(state) == {volume.id: ["volume"]}


def test_a_reshaped_array_is_sent():
    points = k3d.points(np.zeros((10, 3), np.float32))
    _, state = _synced(points)

    points.positions = np.zeros((4, 3), np.float32)

    assert _touched(state) == {points.id: ["positions"]}


def test_shape_and_dtype_are_part_of_the_snapshot():
    # the Array traits coerce, so this is below the trait layer: same bytes is not enough
    assert _snapshot(np.zeros((4, 4), np.float32)) != _snapshot(np.zeros((2, 8), np.float32))
    assert _snapshot(np.zeros(4, np.float32)) != _snapshot(np.zeros(4, np.int32))


def test_a_scalar_change_is_sent():
    points = k3d.points(np.zeros((10, 3), np.float32), point_size=0.1)
    _, state = _synced(points)

    points.point_size = 0.2

    assert _touched(state) == {points.id: ["point_size"]}


def test_a_strided_array_still_diffs():
    # a view cannot be checksummed through its buffer, so it falls back to a deepcopy
    origins = np.zeros((20, 3), np.float32)[::2]
    vectors = k3d.vectors(origins, np.zeros((10, 3), np.float32))
    _, state = _synced(vectors)

    assert _touched(state) == {}

    vectors.origins[0, 0] = 1.0

    assert _touched(state) == {vectors.id: ["origins"]}


def test_a_time_series_diffs_frame_by_frame():
    line = k3d.line({"0": VERTICES, "1": VERTICES * 2}, width=0.1)
    _, state = _synced(line)

    assert _touched(state) == {}

    line.vertices["1"][0, 0] = 9.0

    assert _touched(state) == {line.id: ["vertices"]}


def test_a_new_time_series_frame_is_sent():
    line = k3d.line({"0": VERTICES}, width=0.1)
    _, state = _synced(line)

    line.vertices["1"] = VERTICES * 2

    assert _touched(state) == {line.id: ["vertices"]}


def test_a_removed_object_is_sent_as_none():
    mesh = k3d.mesh(VERTICES, INDICES)
    plot, state = _synced(mesh, k3d.points(np.zeros((4, 3), np.float32)))

    plot -= mesh

    assert state.diff()["objects_diff"][mesh.id] is None
    assert mesh.id not in state.synced_objects


def test_an_added_object_is_sent_whole():
    plot, state = _synced(k3d.mesh(VERTICES, INDICES))
    points = k3d.points(np.zeros((4, 3), np.float32))
    plot += points

    sent = state.diff()["objects_diff"]

    assert set(sent) == {points.id}
    assert set(sent[points.id]) == set(points._synced_props)


def test_the_plot_diff_only_carries_what_moved():
    plot, state = _synced(k3d.mesh(VERTICES, INDICES))

    plot.camera = [4, 0, 0, 0, 0, 0, 0, 0, 1]

    assert list(state.diff()["plot_diff"]) == ["camera"]


def test_the_snapshot_does_not_keep_the_array_alive():
    # the deepcopy it replaces doubled the resident cost of every volume on the plot
    data = np.zeros((64, 64, 64), np.float32)
    snapshot = _snapshot(data)
    alive = weakref.ref(data)

    del data
    gc.collect()

    assert isinstance(snapshot, _ArrayFingerprint)
    assert alive() is None


def test_a_volume_is_snapshotted_as_a_fingerprint():
    volume = k3d.volume(np.zeros((16, 16, 16), np.float32), color_range=[0, 1])
    _, state = _synced(volume)

    assert isinstance(state.synced_objects[volume.id]["volume"], _ArrayFingerprint)


def test_the_checksum_compares_bytes_not_values():
    # two deliberate divergences from the elementwise != this replaced, pinned so that
    # changing either is a decision: nan equals itself, and -0.0 does not equal 0.0
    nan = np.array([1.0, np.nan], np.float32)

    assert _snapshot(nan) == _snapshot(nan)
    assert _snapshot(np.array([0.0], np.float32)) != _snapshot(np.array([-0.0], np.float32))
