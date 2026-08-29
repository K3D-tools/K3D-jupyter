"""glTF export has to be honest about what it can carry.

The format stores triangles and PBR materials, so an object whose shape only exists while its
shader runs - a ray-marched volume, a billboarded point, a line the vertex stage widens into a
ribbon - has no surface to hand over. Exporting its buffers anyway would ship a solid that looks
nothing like the render, so the exporter hides those objects for the duration of the parse. That
hiding, and the naming that goes with it, mutates the live scene and has to be undone afterwards.
"""

import json
import struct

import numpy as np
import pytest

import k3d

from .plot_compare import prepare

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A

POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def _document(blob):
    """Return the JSON chunk of a .glb, checking the container along the way."""
    magic, version, length = struct.unpack_from("<III", blob, 0)

    assert magic == GLB_MAGIC
    assert version == 2
    assert length == len(blob)

    offset = 12
    document = None

    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", blob, offset)
        offset += 8

        if chunk_type == JSON_CHUNK:
            document = json.loads(blob[offset:offset + chunk_length])

        offset += chunk_length

    assert document is not None

    return document


def _mesh_names(document):
    """Names of every node that actually carries geometry."""
    return {
        node["name"]
        for node in document.get("nodes", [])
        if node.get("mesh") is not None and node.get("name")
    }


def _export():
    pytest.headless.sync(hold_until_refreshed=True)

    return _document(pytest.headless.get_gltf())


def test_exports_geometry_and_skips_shader_only_objects():
    prepare()

    pytest.plot += k3d.mesh(VERTICES, INDICES, name="triangle")
    pytest.plot += k3d.points(POSITIONS, point_size=0.2, shader="mesh", name="spheres")
    pytest.plot += k3d.points(POSITIONS, point_size=0.2, shader="3d", name="billboards")
    pytest.plot += k3d.line(POSITIONS, width=0.1, shader="thick", name="ribbon")
    pytest.plot += k3d.volume(
        np.random.random((8, 8, 8)).astype(np.float32), name="cloud"
    )

    names = _mesh_names(_export())

    assert {"triangle", "spheres"} <= names
    assert names.isdisjoint({"billboards", "ribbon", "cloud"})


def test_exported_mesh_keeps_its_vertices():
    """A node in the file is worth nothing if the buffer behind it is empty."""
    prepare()

    pytest.plot += k3d.mesh(VERTICES, INDICES, name="triangle")

    document = _export()
    node = next(n for n in document["nodes"] if n.get("name") == "triangle")
    primitive = document["meshes"][node["mesh"]]["primitives"][0]

    assert document["accessors"][primitive["attributes"]["POSITION"]]["count"] == len(VERTICES)
    assert document["accessors"][primitive["indices"]]["count"] == INDICES.size


def test_export_leaves_the_scene_untouched():
    """Hiding and renaming happen on the live objects, so a failed restore is a visible bug."""
    prepare()

    pytest.plot += k3d.points(POSITIONS, point_size=0.2, shader="3d", name="billboards")
    pytest.plot += k3d.mesh(VERTICES, INDICES, name="triangle")

    probe = """
    var world = K3DInstance.getWorld();
    var state = [];

    world.K3DObjects.traverse(function (o) {
        if (o.isMesh || o.isLine || o.isPoints) { state.push([o.visible, o.name]); }
    });

    return state;
    """

    pytest.headless.sync(hold_until_refreshed=True)
    before = pytest.headless.browser.execute_script(probe)

    pytest.headless.get_gltf()

    assert pytest.headless.browser.execute_script(probe) == before


def test_dom_backed_objects_leave_nothing_behind():
    """A label keeps a stub line for its leader, which would export as a segment with no text."""
    prepare()

    pytest.plot += k3d.label("caption", name="caption")
    pytest.plot += k3d.texture_text("caption", name="billboard")

    document = _export()

    assert not [node for node in document.get("nodes", []) if node.get("mesh") is not None]


def test_hidden_objects_stay_out_of_the_export():
    prepare()

    hidden = k3d.mesh(VERTICES, INDICES, name="triangle")
    hidden.visible = False

    pytest.plot += hidden

    assert "triangle" not in _mesh_names(_export())


def test_what_cannot_be_exported_is_reported():
    """A scene of volumes alone produces an empty file, which needs saying out loud."""
    prepare()

    pytest.plot += k3d.volume(np.random.random((8, 8, 8)).astype(np.float32), name="cloud")
    pytest.plot += k3d.mesh(VERTICES, INDICES, name="triangle")
    pytest.headless.sync(hold_until_refreshed=True)

    warnings = pytest.headless.browser.execute_script(
        """
    window.__warnings = [];

    var original = console.warn;

    console.warn = function () {
        window.__warnings.push(Array.prototype.join.call(arguments, ' '));
        original.apply(console, arguments);
    };

    return K3DInstance.getGLTF().then(function () {
        console.warn = original;

        return window.__warnings;
    });
    """
    )

    assert any("cloud" in w for w in warnings)
    assert not any("triangle" in w for w in warnings)


def test_the_gui_offers_the_export():
    """The button is the only route to glTF for a snapshot, which has no kernel behind it."""
    prepare()
    pytest.headless.sync(hold_until_refreshed=True)

    names = pytest.headless.browser.execute_script(
        "return K3DInstance.gui.controllersRecursive().map(function (c) { return c._name; });"
    )

    assert "Export glTF" in names
