"""Hiding the menu, which is what the headless driver does before any object is loaded."""

import pytest

HIDE = """
const done = arguments[arguments.length - 1];

require(['k3d'], (lib) => {
    const target = document.createElement('div');
    target.style.width = '320px';
    target.style.height = '240px';
    document.body.appendChild(target);

    const instance = new lib.K3D(lib.ThreeJsProvider, target, {});
    const before = !!instance.gui;
    let error = null;

    try {
        instance.setMenuVisibility(false);
    } catch (e) {
        error = e.toString();
    }

    const after = !!instance.gui;

    instance.disable();
    target.remove();

    done({error: error, before: before, after: after});
});
"""


def test_hides_on_a_plot_with_no_objects():
    # gui_map only exists once objectsGUIprovider has seen an object, and the headless driver
    # applies the plot parameters before it loads any.
    result = pytest.headless.browser.execute_async_script(HIDE)

    assert result["error"] is None, result["error"]
    assert result["before"] is True, "the menu was never built, so nothing was torn down"
    assert result["after"] is False
