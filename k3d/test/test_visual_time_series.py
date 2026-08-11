"""Visual coverage for stepped time series playback and for frames of unequal size.

Both stepped tests compare against the reference of the keyframe they must be holding, so a
value that gets blended anyway fails rather than silently drifting.

The unequal-size case is the one that used to be wrong in a way nothing reported: the blend
produced a buffer as long as the larger frame while shape still described the smaller one.

The first two points are identical in every frame and pin the bounding box; only the rest move,
so the camera set up by prepare() frames every state the same way. camera_factor=None keeps
compare() from re-fitting, which is path dependent through grid auto-fit and makes the same
scene render differently depending on which test ran before.
"""

import numpy as np
import pytest

import k3d
from .plot_compare import compare, prepare

ANCHORS = [[-1, -1, -1], [1, 1, 1]]

POSITIONS_A = np.array(
    ANCHORS + [[-0.7, 0, 0], [0, -0.7, 0]],
    dtype=np.float32,
)
POSITIONS_B = np.array(
    ANCHORS + [[0.7, 0, 0], [0, 0.7, 0]],
    dtype=np.float32,
)

# Five movers against two: no per-point correspondence exists, so no blend is meaningful.
POSITIONS_B_LARGER = np.array(
    ANCHORS + [
        [0.7, 0, 0], [0, 0.7, 0], [0.7, 0.7, 0], [0, 0.7, 0.7], [0.7, 0, 0.7],
    ],
    dtype=np.float32,
)


def _points(frame_b):
    return k3d.points(
        {"0.0": POSITIONS_A, "1.0": frame_b},
        point_size=0.25,
        color=0x3F6BC5,
    )


def test_time_series_interpolates():
    prepare()
    pytest.plot += _points(POSITIONS_B)

    # Rendered once first: time set in the same sync that brings the object in does not take.
    compare("time_series_frame_a", camera_factor=None)

    pytest.plot.time = 0.5

    compare("time_series_interpolated", camera_factor=None)


def test_time_series_holds_keyframe():
    prepare()
    pytest.plot += _points(POSITIONS_B)

    compare("time_series_frame_a", camera_factor=None)

    pytest.plot.time_interpolation = False
    pytest.plot.time = 0.5

    # Same reference as time 0.0: holding means the keyframe itself, not something near it.
    compare("time_series_frame_a", camera_factor=None)


def test_time_series_unequal_frame_sizes():
    prepare()
    pytest.plot += _points(POSITIONS_B_LARGER)

    compare("time_series_unequal_frame_a", camera_factor=None)

    pytest.plot.time = 0.5

    # Blending is undefined here, so the nearer keyframe is shown whole.
    compare("time_series_unequal_frame_a", camera_factor=None)
