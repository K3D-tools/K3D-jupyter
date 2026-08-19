"""The renderer trait chain, without images: degradation and state hygiene."""

import numpy as np
import pytest

import k3d
from .plot_compare import prepare


def test_unknown_renderer_degrades_to_simple():
    prepare()

    pytest.headless.browser.execute_script("K3DInstance.setRenderer('pathtraced')")
    mode = pytest.headless.browser.execute_script("return K3DInstance.parameters.renderer")

    assert mode == "simple"

    # the browser-side call above is invisible to the sync diff; realign before the next test
    pytest.headless.browser.execute_script("K3DInstance.setRenderer('simple')")


def test_cinematic_is_a_legal_renderer():
    prepare()

    # the headless container has WebGL2 + float textures, so the switch must commit
    mode = pytest.headless.browser.execute_script("""
    K3DInstance.setRenderer('cinematic');
    const switched = K3DInstance.parameters.renderer;
    K3DInstance.setRenderer('simple');
    return [switched, K3DInstance.parameters.renderer];
    """)

    assert mode == ["cinematic", "simple"]


def test_switching_modes_leaves_no_state():
    prepare()

    state = pytest.headless.browser.execute_script("""
    const w = K3DInstance.getWorld();
    K3DInstance.setRenderer('advanced');
    const advanced = {env: w.scene.environment !== null, head: w.headLight.visible};
    K3DInstance.setRenderer('simple');
    const back = {env: w.scene.environment !== null, bg: w.scene.background !== null,
                  head: w.headLight.visible};
    return {advanced: advanced, back: back};
    """)

    assert state["advanced"]["env"] is True
    assert state["advanced"]["head"] is False
    assert state["back"]["env"] is False
    assert state["back"]["bg"] is False
    assert state["back"]["head"] is True


def test_environment_catalog_resolves_python_side():
    plot = k3d.plot()
    plot.environment = "autoshop_01"

    assert isinstance(plot.environment, np.ndarray)
    assert plot.environment.shape == (256, 512, 3)

    plot.environment = "neutral"
    assert plot.environment == "neutral"
