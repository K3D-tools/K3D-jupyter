"""Point picking has to survive the trip from Python down to the raycaster.

The browser arms a cloud only when the object JSON carries a callback flag, and the pick
radius has to follow point_size - the billboard shader keeps it on the material, the mesh
shader in the instance scale on top of the icosahedron it bakes point_size into. None of
this was reachable from Python until Points grew the callback traits, so nothing exercised
either path.
"""

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

# aim the raycaster dead centre at one point and report what the object hands back
PROBE = """
var world = K3DInstance.getWorld();
var index = arguments[0];
var obj = null;

world.K3DObjects.traverse(function (o) {
    if (o.interactions && obj === null) { obj = o; }
});

if (obj === null) {
    return {armed: false};
}

var v = obj.position.clone();

// the mesh shader instances one icosahedron, so its positions live in instanceMatrix
if (obj.isInstancedMesh) {
    var m = obj.instanceMatrix.array;
    v.set(m[index * 16 + 12], m[index * 16 + 13], m[index * 16 + 14]);
} else {
    var p = obj.geometry.attributes.position.array;
    v.set(p[index * 3], p[index * 3 + 1], p[index * 3 + 2]);
}

v.applyMatrix4(obj.matrixWorld);
v.project(world.camera);

world.raycaster.setFromCamera({x: v.x, y: v.y}, world.camera);

var hits = obj.interactions.intersect(world.raycaster);

return {armed: true, hits: hits.length, index: hits.length ? hits[0].index : null};
"""


def _probe(index):
    return pytest.headless.browser.execute_script(PROBE, index)


def _cloud(shader, callback=True):
    prepare()

    obj = k3d.points(POSITIONS, point_size=0.2, shader=shader)

    if callback:
        obj.click_callback = lambda params: None

    pytest.plot += obj
    pytest.headless.sync(hold_until_refreshed=True)

    return obj


@pytest.mark.parametrize("shader", ["3d", "mesh"])
def test_every_point_picks_itself(shader):
    """A radius wider than the points would keep returning whichever one is nearest the camera."""
    _cloud(shader)

    assert [_probe(i)["index"] for i in range(len(POSITIONS))] == list(range(len(POSITIONS)))


def test_callbacks_arm_and_disarm_picking():
    obj = _cloud("mesh", callback=False)
    assert _probe(0) == {"armed": False}

    obj.click_callback = lambda params: None
    pytest.headless.sync(hold_until_refreshed=True)
    assert _probe(0)["hits"] == 1

    obj.click_callback = None
    pytest.headless.sync(hold_until_refreshed=True)
    assert _probe(0) == {"armed": False}


def test_switching_shader_keeps_picking_armed():
    """Changing the shader swaps the loader, and the rebuilt object has to arm itself again."""
    obj = _cloud("3d")
    assert _probe(2)["index"] == 2

    obj.shader = "mesh"
    pytest.headless.sync(hold_until_refreshed=True)

    assert _probe(2)["index"] == 2
