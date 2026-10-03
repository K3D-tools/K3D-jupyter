"""The panel offers width for thick and mesh lines and radial_segments for mesh, following the shader."""

import pytest

PROBE = """
const done = arguments[arguments.length - 1];

require(['k3d'], (lib) => {
    const target = document.createElement('div');
    target.style.width = '320px';
    target.style.height = '240px';
    document.body.appendChild(target);

    const instance = new lib.K3D(lib.ThreeJsProvider, target, {});
    const seen = {};
    let error = null;

    // controllersMap keys are exactly the controls on screen
    function controls() {
        return Object.keys(instance.gui_map[1].controllersMap);
    }

    // exactly what lil-gui does: write the value onto the json, then reload with the change set
    function setShader(shader) {
        const json = instance.getWorld().ObjectsListJson[1];

        json.shader = shader;

        return instance.reload(json, { shader: shader });
    }

    (async () => {
        try {
            await instance.load({ objects: [{
                id: 1,
                type: 'Line',
                shader: 'simple',
                visible: true,
                width: 0.1,
                radial_segments: 8,
                color: 255,
                vertices: { data: new Float32Array([0, 0, 0, 1, 1, 1]), shape: [2, 3] },
                model_matrix: { data: new Float32Array(
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), shape: [4, 4] },
            }] });

            seen.simple = controls();

            await setShader('thick');
            seen.thick = controls();

            await setShader('mesh');
            seen.mesh = controls();

            await setShader('simple');
            seen.back = controls();
        } catch (e) {
            error = e.toString();
        }

        instance.disable();
        target.remove();

        done({ error: error, seen: seen });
    })();
});
"""


@pytest.fixture(scope="module")
def controls():
    result = pytest.headless.browser.execute_async_script(PROBE)

    assert result["error"] is None, result["error"]

    return result["seen"]


def test_simple_offers_neither(controls):
    assert "width" not in controls["simple"]
    assert "radial_segments" not in controls["simple"]


def test_thick_offers_width_only(controls):
    # MeshLineMaterial.lineWidth takes it; there is no tube, so no cross-section to set
    assert "width" in controls["thick"]
    assert "radial_segments" not in controls["thick"]


def test_mesh_offers_both(controls):
    assert "width" in controls["mesh"]
    assert "radial_segments" in controls["mesh"]


def test_going_back_to_simple_takes_them_away(controls):
    # a control left behind would write a parameter the renderer no longer reads
    assert "width" not in controls["back"]
    assert "radial_segments" not in controls["back"]
