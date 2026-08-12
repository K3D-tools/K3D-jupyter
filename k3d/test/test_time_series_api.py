"""Keyframe introspection and frame stepping, kernel side, without a browser."""

import numpy as np
import pytest

import k3d

A = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _plot():
    plot = k3d.plot()
    plot += k3d.points({"0.0": A, "0.5": A * 2, "2.0": A * 3})

    return plot


def test_times_are_the_union_over_objects():
    plot = _plot()
    plot += k3d.points({"1.0": A})

    assert plot.get_time_series_times() == [0.0, 0.5, 1.0, 2.0]


def test_times_ignore_plain_dict_traits():
    plot = k3d.plot()
    plot += k3d.points(A, custom_data={"7.0": 1})

    assert plot.get_time_series_times() == []


def test_times_include_camera_animation():
    plot = k3d.plot()
    plot.camera_animation = {"3.0": [1, 2, 3, 0, 0, 0, 0, 0, 1]}

    assert plot.get_time_series_times() == [3.0]


def test_range_always_contains_zero():
    plot = k3d.plot()
    plot += k3d.points({"4.0": A, "6.0": A * 2})

    assert plot.get_time_series_range() == [0.0, 6.0]


def test_range_of_an_empty_plot():
    assert k3d.plot().get_time_series_range() == [0.0, 0.0]


def test_stepping_starts_from_the_nearest_keyframe():
    plot = _plot()
    plot.time = 0.6

    assert plot.next_frame() == 2.0
    assert plot.previous_frame() == 0.5
    assert plot.previous_frame() == 0.0


def test_stepping_clamps_at_both_ends():
    plot = _plot()

    assert plot.previous_frame() == 0.0

    plot.time = 2.0

    assert plot.next_frame() == 2.0


def test_stepping_without_a_time_series_is_a_noop():
    plot = k3d.plot()
    plot += k3d.points(A)
    plot.time = 0.25

    assert plot.next_frame() == 0.25
    assert plot.time == 0.25


@pytest.mark.parametrize("step", [2, -2])
def test_stepping_by_more_than_one(step):
    plot = _plot()
    plot.time = 0.5

    expected = 2.0 if step > 0 else 0.0

    assert plot.step_frame(step) == expected
