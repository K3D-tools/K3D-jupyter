"""Regression: a GUI visible toggle echoed by the widget bridge must not orphan objects.

Reproduces the JupyterLab pathology at the core level: every checkbox click runs
K3D.reload twice in one tick (changeParameter + the model echo). Before the fix the
second re-show create could not see the first one (ObjectsById was populated only in
reload's .then), so every show leaked an unremovable duplicate into the scene.
"""

import numpy as np
import pytest
import k3d
from .plot_compare import prepare

# lil-gui writes the value onto the bound json before onChange, then changeParameter
# reloads and dispatches OBJECT_CHANGE; the anywidget bridge echoes a second reload
# with the same (shared) json in the same tick
DOUBLE_TOGGLE = """
var id = arguments[0], value = arguments[1];
var json = K3DInstance.getWorld().ObjectsListJson[id];
json.visible = value;
K3DInstance.reload(json, {visible: value}, false);
K3DInstance.reload(json, {visible: value}, false);
"""

CENSUS = """
var w = K3DInstance.getWorld();
var visible = 0;
w.K3DObjects.children.forEach(function (o) { if (o.visible) { visible += 1; } });
return {children: w.K3DObjects.children.length, visible: visible};
"""

SETTLE = "K3DInstance.render(); K3DInstance.dispatch(K3DInstance.events.RENDERED)"


def _scene():
    g = np.linspace(-1, 1, 24, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    vol = (np.exp(-(x ** 2 + y ** 2 + z ** 2) / 0.4) * 800).astype(np.float32)

    volume = k3d.volume(vol, samples=64, alpha_coef=100)
    mesh = k3d.mesh(
        np.array([[0, 2, 0], [2, 2, 0], [2, 0, 0], [0, 0, 2]], np.float32),
        np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], np.uint32),
        color=0xFF8800,
    )

    pytest.plot += volume
    pytest.plot += mesh
    pytest.plot.grid_visible = False
    pytest.headless.sync(hold_until_refreshed=True)
    pytest.headless.camera_reset(1.0)

    return volume, mesh


def _toggle(obj_id, value):
    pytest.headless.browser.execute_script(DOUBLE_TOGGLE, obj_id, value)


def _census():
    pytest.headless.browser.execute_script(SETTLE)
    return pytest.headless.browser.execute_script(CENSUS)


def test_gui_visible_toggle_leaves_no_orphans():
    prepare()
    volume, mesh = _scene()

    assert _census() == {"children": 2, "visible": 2}

    for obj in (volume, mesh):
        for _ in range(3):
            _toggle(obj.id, False)
            state = _census()
            assert state["visible"] == 1, state
            _toggle(obj.id, True)
            state = _census()
            assert state["children"] == 2, state
            assert state["visible"] == 2, state


def test_rapid_visible_toggle_ends_hidden_and_clean():
    prepare()
    volume, mesh = _scene()

    # like clicking faster than creates resolve: no settle between reloads
    for _ in range(4):
        for obj in (volume, mesh):
            _toggle(obj.id, False)
            _toggle(obj.id, True)
            _toggle(obj.id, False)

    state = _census()
    assert state["children"] <= 2, state
    assert state["visible"] == 0, state

    for obj in (volume, mesh):
        _toggle(obj.id, True)

    state = _census()
    assert state == {"children": 2, "visible": 2}

    pytest.plot -= volume
    pytest.plot -= mesh
    pytest.headless.sync(hold_until_refreshed=True)

    state = _census()
    assert state == {"children": 0, "visible": 0}
