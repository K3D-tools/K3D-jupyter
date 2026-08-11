"""The frontend side of the time series extension points.

TIME_CHANGE must not travel on PARAMETERS_CHANGE: that channel is written back to the widget
model, so a time set from the kernel would be echoed straight back.
"""

import numpy as np
import pytest

import k3d
from .plot_compare import prepare

A = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
FRAMES = {"0.0": A, "0.5": A * 0.5, "2.0": A * 0.25}

LISTEN = """
const done = arguments[arguments.length - 1];

const times = [];
const autoPlay = [];
const parameters = [];

const timeId = K3DInstance.on(K3DInstance.events.TIME_CHANGE, (t) => times.push(t));
const playId = K3DInstance.on(K3DInstance.events.AUTO_PLAY_CHANGE, (s) => autoPlay.push(s));
const paramId = K3DInstance.on(K3DInstance.events.PARAMETERS_CHANGE, (c) => parameters.push(c.key));

Promise.resolve(K3DInstance.setTime(0.5)).then(() => {
    K3DInstance.startAutoPlay();
    K3DInstance.stopAutoPlay();

    K3DInstance.off(K3DInstance.events.TIME_CHANGE, timeId);
    K3DInstance.off(K3DInstance.events.AUTO_PLAY_CHANGE, playId);
    K3DInstance.off(K3DInstance.events.PARAMETERS_CHANGE, paramId);

    done({times: times, autoPlay: autoPlay, parameters: parameters});
});
"""

STEP = """
const done = arguments[arguments.length - 1];

K3DInstance.setTime(0.0);

Promise.resolve().then(() => {
    const forward = K3DInstance.stepFrame(1);
    const back = K3DInstance.stepFrame(-1);
    const clamped = K3DInstance.stepFrame(-1);

    done({info: K3DInstance.getTimeSeriesInfo(), forward: forward, back: back, clamped: clamped});
});
"""


def _animated_plot():
    prepare()
    pytest.plot += k3d.points(FRAMES, point_size=0.2)
    pytest.headless.sync(hold_until_refreshed=True)


def test_time_and_auto_play_are_announced():
    _animated_plot()

    result = pytest.headless.browser.execute_async_script(LISTEN)

    assert result["times"] == [0.5]
    assert result["autoPlay"] == [True, False]

    assert "time" not in result["parameters"], (
        "time reached PARAMETERS_CHANGE, which is echoed back to the model: %s"
        % result["parameters"]
    )


def test_range_and_times_match_the_kernel():
    _animated_plot()

    result = pytest.headless.browser.execute_async_script(STEP)

    assert result["info"]["times"] == pytest.plot.get_time_series_times()
    assert [result["info"]["min"], result["info"]["max"]] == pytest.plot.get_time_series_range()


def test_frame_stepping_matches_the_kernel():
    _animated_plot()

    result = pytest.headless.browser.execute_async_script(STEP)

    assert result["forward"] == 0.5
    assert result["back"] == 0.0
    assert result["clamped"] == 0.0


SNAP = """
const done = arguments[arguments.length - 1];

const controller = K3DInstance.GUI.controls.controllersMap.time;

if (!controller) {
    done({error: 'no time controller'});
} else {
    controller.setValue(0.4);

    Promise.resolve().then(() => {
        done({time: K3DInstance.parameters.time});
    });
}
"""


def test_slider_snaps_to_a_keyframe_only_when_stepping():
    _animated_plot()

    interpolated = pytest.headless.browser.execute_async_script(SNAP)
    assert "error" not in interpolated, interpolated.get("error")
    assert interpolated["time"] == 0.4

    pytest.plot.time_interpolation = False
    pytest.headless.sync(hold_until_refreshed=True)

    stepped = pytest.headless.browser.execute_async_script(SNAP)
    assert stepped["time"] == 0.5
