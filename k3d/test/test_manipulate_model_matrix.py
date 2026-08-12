"""The manipulator has to report the transform it applied back to the model.

The write-back cannot be driven by a real drag here, so the tests move the object and fire the
control's dragging-changed pair directly, which is what the handler listens to.

The wire format is row-major (commonUpdate feeds it straight into Matrix4.set), while
Matrix4.elements is column-major, so a missing transpose is caught by the translation showing
up in the bottom row instead of the last column.

The second test covers the feedback question: the kernel echoes model_matrix back, so the value
returns through the regular update path. It must neither move the object nor produce another
change, otherwise one drag would loop.
"""

import numpy as np
import pytest

from .plot_compare import prepare

import k3d

CAPTURE = """
const object = K3DInstance.getWorld().K3DObjects.children[0];

if (!object.transformControls) {
    return {error: 'no transformControls, manipulate mode did not attach'};
}

const captured = [];
const listener = K3DInstance.on(K3DInstance.events.OBJECT_CHANGE, (change) => {
    captured.push({
        key: change.key,
        data: Array.from(change.value.data),
        shape: change.value.shape,
    });
});

object.position.set(1, 2, 3);
object.updateMatrix();

object.transformControls.dispatchEvent({type: 'dragging-changed', value: true});
object.transformControls.dispatchEvent({type: 'dragging-changed', value: false});

K3DInstance.off(K3DInstance.events.OBJECT_CHANGE, listener);

return {captured: captured};
"""

ECHO = """
const done = arguments[arguments.length - 1];
const world = K3DInstance.getWorld();
const object = world.K3DObjects.children[0];

const captured = [];
const listener = K3DInstance.on(K3DInstance.events.OBJECT_CHANGE, (change) => {
    captured.push(change);
});

object.position.set(1, 2, 3);
object.updateMatrix();

object.transformControls.dispatchEvent({type: 'dragging-changed', value: true});
object.transformControls.dispatchEvent({type: 'dragging-changed', value: false});

const dispatchedByDrag = captured.length;
const afterDrag = Array.from(object.matrix.elements);

// What the kernel sends back after saving the trait: the same value, through the update path.
const json = world.ObjectsListJson[object.K3DIdentifier];
const echoed = {data: new Float32Array(captured[0].value.data), shape: [4, 4]};

json.model_matrix = echoed;

Promise.resolve(K3DInstance.reload(json, {model_matrix: echoed})).then(() => {
    K3DInstance.off(K3DInstance.events.OBJECT_CHANGE, listener);

    done({
        dispatchedByDrag: dispatchedByDrag,
        dispatchedTotal: captured.length,
        afterDrag: afterDrag,
        afterEcho: Array.from(world.K3DObjects.children[0].matrix.elements),
    });
});
"""


def _manipulable_triangle():
    prepare()

    pytest.plot += k3d.mesh(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
    )
    pytest.headless.sync(hold_until_refreshed=True)

    # The mode trait is not among the parameters headless sends, so the view mode is switched in
    # the page. It attaches a control to every object already in the scene.
    pytest.headless.browser.execute_script("K3DInstance.setViewMode('manipulate');")


def test_manipulator_reports_model_matrix():
    _manipulable_triangle()

    result = pytest.headless.browser.execute_script(CAPTURE)

    assert "error" not in result, result.get("error")

    changes = [c for c in result["captured"] if c["key"] == "model_matrix"]
    assert len(changes) == 1, "expected exactly one model_matrix change, got %d" % len(
        changes
    )

    assert changes[0]["shape"] == [4, 4]

    matrix = np.array(changes[0]["data"], dtype=np.float32).reshape(4, 4)

    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_model_matrix_echo_does_not_loop():
    _manipulable_triangle()

    result = pytest.headless.browser.execute_async_script(ECHO)

    assert result["dispatchedByDrag"] == 1
    assert result["dispatchedTotal"] == 1, (
        "the echoed model_matrix produced %d further change(s), which would loop"
        % (result["dispatchedTotal"] - 1)
    )
    np.testing.assert_allclose(result["afterEcho"], result["afterDrag"], atol=1e-6)
