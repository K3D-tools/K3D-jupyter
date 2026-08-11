"""A plot constructed against a node that is not in the document yet, as Panel does."""

import pytest

DETACHED = """
const done = arguments[arguments.length - 1];

require(['k3d'], (lib) => {
    const target = document.createElement('div');
    target.style.width = '320px';
    target.style.height = '240px';

    let instance = null;

    try {
        instance = new lib.K3D(lib.ThreeJsProvider, target, {menuVisibility: false});
    } catch (e) {
        done({error: e.toString()});
        return;
    }

    const detached = {disabling: !!instance.disabling, width: instance.getWorld().width};

    document.body.appendChild(target);

    setTimeout(() => {
        const attached = {disabling: !!instance.disabling, width: instance.getWorld().width};

        instance.disable();
        target.remove();

        done({detached: detached, attached: attached});
    }, 300);
});
"""


def test_survives_construction_outside_the_document():
    result = pytest.headless.browser.execute_async_script(DETACHED)

    assert "error" not in result, result.get("error")
    assert result["detached"]["disabling"] is False, (
        "the instance tore itself down during construction"
    )


def test_starts_rendering_once_inserted():
    result = pytest.headless.browser.execute_async_script(DETACHED)

    assert "error" not in result, result.get("error")

    # 300 is the canvas fallback for a node that measures zero, 320 the container once inserted.
    assert result["detached"]["width"] == 300
    assert result["attached"]["width"] == 320


REMOVED = """
const done = arguments[arguments.length - 1];

require(['k3d'], (lib) => {
    const target = document.createElement('div');
    target.style.width = '320px';
    target.style.height = '240px';
    document.body.appendChild(target);

    const instance = new lib.K3D(lib.ThreeJsProvider, target, {menuVisibility: false});

    setTimeout(() => {
        target.remove();

        setTimeout(() => {
            const disabling = !!instance.disabling;

            if (!disabling) {
                instance.disable();
            }

            done({disabling: disabling});
        }, 300);
    }, 300);
});
"""


def test_still_tears_down_when_the_node_goes_away():
    result = pytest.headless.browser.execute_async_script(REMOVED)

    assert result["disabling"] is True, (
        "a node that was attached and then removed no longer tears the instance down"
    )
