"""k3d.glb() and k3d.gltf(): what a glTF file becomes, checked without a browser.

Most documents here are built by the tests themselves, so each one holds exactly the feature
under test and its expected values are known by construction. The Khronos sample models in
assets/gltf are there for what a hand-built file would only imitate.
"""

import base64
import io
import json
import os
import struct
import warnings

import numpy as np
import pytest

import k3d
from k3d.objects import Group, Lines, Mesh, Points

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gltf")

# a 1x1 PNG and a 1x1 JPEG - the reader passes images through, it never decodes them
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")

FLOAT, UBYTE, USHORT, UINT, SHORT = 5126, 5121, 5123, 5125, 5122


class Document:
    """A minimal glTF writer: buffers, views, accessors, and the JSON to put them in."""

    def __init__(self):
        self.json = {"asset": {"version": "2.0"}, "buffers": [], "bufferViews": [], "accessors": [],
                     "meshes": [], "nodes": [], "materials": [], "scenes": [{"nodes": []}], "scene": 0}
        self.bin = bytearray()

    def view(self, data, stride=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if stride:
            view["byteStride"] = stride
        self.bin += data
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def accessor(self, array, component=FLOAT, kind=None, normalized=False, view=None, offset=0):
        array = np.asarray(array)
        dtype = {FLOAT: np.float32, UBYTE: np.uint8, USHORT: np.uint16, UINT: np.uint32,
                 SHORT: np.int16}[component]
        if kind is None:
            kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[1 if array.ndim == 1 else array.shape[1]]
        if view is None:
            view = self.view(array.astype(dtype).tobytes())
        accessor = {"bufferView": view, "byteOffset": offset, "componentType": component,
                    "count": len(array), "type": kind}
        if normalized:
            accessor["normalized"] = True
        self.json["accessors"].append(accessor)
        return len(self.json["accessors"]) - 1

    def image(self, data, mime="image/png"):
        self.json.setdefault("images", []).append({"bufferView": self.view(data), "mimeType": mime})
        self.json.setdefault("textures", []).append({"source": len(self.json["images"]) - 1})
        return len(self.json["textures"]) - 1

    def material(self, **definition):
        self.json["materials"].append(definition)
        return len(self.json["materials"]) - 1

    def mesh(self, primitives, name=None, node=None):
        self.json["meshes"].append({"primitives": primitives})
        node = dict(node or {})
        node["mesh"] = len(self.json["meshes"]) - 1
        if name:
            node["name"] = name
        return self.node(node)

    def node(self, node, root=True):
        self.json["nodes"].append(node)
        index = len(self.json["nodes"]) - 1
        if root:
            self.json["scenes"][0]["nodes"].append(index)
        return index

    def triangle(self, **attributes):
        """A one-triangle primitive, with whatever extra attributes and keys are given."""
        primitive = {"attributes": {"POSITION": self.accessor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])}}
        for key, value in attributes.items():
            if key.isupper():
                primitive["attributes"][key] = value
            else:
                primitive[key] = value
        return primitive

    def glb(self):
        self.json["buffers"] = [{"byteLength": len(self.bin)}] if self.bin else []
        text = json.dumps(self.json).encode()
        text += b" " * (-len(text) % 4)
        binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
        if binary:
            chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
        return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks

    def gltf(self, uri=None):
        """JSON with the buffer as a data: URI, or as `uri` (the caller writes that file)."""
        document = dict(self.json)
        document["buffers"] = [{"byteLength": len(self.bin),
                                "uri": uri or "data:application/octet-stream;base64,"
                                + base64.b64encode(bytes(self.bin)).decode()}]
        return json.dumps(document).encode()


def read(doc, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return k3d.glb(doc.glb(), **kwargs)


def only(group):
    objects = list(group)
    assert len(objects) == 1, objects
    return objects[0]


# -- containers ---------------------------------------------------------------------------

def test_glb_takes_bytes_a_path_and_a_file_object(tmp_path):
    doc = Document()
    doc.mesh([doc.triangle()], name="tri")
    data = doc.glb()
    path = tmp_path / "model.glb"
    path.write_bytes(data)

    for source in (data, str(path), path, io.BytesIO(data)):
        assert only(k3d.glb(source)).name == "tri"


def test_gltf_with_external_buffer_and_image(tmp_path):
    doc = Document()
    texture = doc.image(PNG)
    doc.json["images"][0] = {"uri": "texture%20file.png"}
    material = doc.material(pbrMetallicRoughness={"baseColorTexture": {"index": texture}})
    uv = doc.accessor([[0, 0], [1, 0], [0, 1]])
    doc.mesh([doc.triangle(TEXCOORD_0=uv, material=material)])
    (tmp_path / "data.bin").write_bytes(bytes(doc.bin))
    (tmp_path / "texture file.png").write_bytes(PNG)
    (tmp_path / "scene.gltf").write_bytes(doc.gltf(uri="data.bin"))

    mesh = only(k3d.gltf(str(tmp_path / "scene.gltf")))

    assert mesh.texture == PNG
    assert mesh.texture_file_format == "png"
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_gltf_bytes_need_everything_inlined():
    doc = Document()
    doc.mesh([doc.triangle()])

    assert only(k3d.gltf(doc.gltf())).type == "Mesh"

    with pytest.raises(ValueError, match="pass the path"):
        k3d.gltf(doc.gltf(uri="elsewhere.bin"))


def test_each_factory_refuses_the_other_format():
    doc = Document()
    doc.mesh([doc.triangle()])

    with pytest.raises(ValueError, match="k3d.gltf"):
        k3d.glb(doc.gltf())
    with pytest.raises(ValueError, match="k3d.glb"):
        k3d.gltf(doc.glb())


def test_glb_and_gltf_of_the_same_model_agree():
    binary = k3d.glb(os.path.join(ASSETS, "BoxTextured.glb"))
    text = k3d.gltf(os.path.join(ASSETS, "BoxTextured", "BoxTextured.gltf"))

    for a, b in zip(binary, text):
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.uvs, b.uvs)
        assert a.texture == b.texture
        np.testing.assert_allclose(a.model_matrix, b.model_matrix)


def test_only_gltf_2_is_read():
    doc = Document()
    doc.json["asset"]["version"] = "1.0"

    with pytest.raises(ValueError, match="2.0"):
        read(doc)


def test_required_extensions_it_cannot_read_are_an_error():
    doc = Document()
    doc.mesh([doc.triangle()])
    doc.json["extensionsRequired"] = ["KHR_draco_mesh_compression"]

    with pytest.raises(NotImplementedError, match="decompress"):
        read(doc)


# -- accessors ----------------------------------------------------------------------------

def test_interleaved_views_are_read_by_their_stride():
    doc = Document()
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    normals = np.array([[0, 0, 1]] * 3, np.float32)
    view = doc.view(np.hstack([positions, normals]).tobytes(), stride=24)
    p = doc.accessor(positions, view=view)
    n = doc.accessor(normals, view=view, offset=12)
    doc.mesh([{"attributes": {"POSITION": p, "NORMAL": n}}])

    mesh = only(read(doc))

    np.testing.assert_array_equal(mesh.vertices, positions)
    np.testing.assert_array_equal(mesh.normals, normals)
    assert mesh.flat_shading is False


def test_normalized_integers_become_floats():
    doc = Document()
    uv = doc.accessor(np.array([[0, 0], [65535, 0], [0, 32768]]), USHORT, normalized=True)
    texture = doc.image(PNG)
    material = doc.material(pbrMetallicRoughness={"baseColorTexture": {"index": texture}})
    doc.mesh([doc.triangle(TEXCOORD_0=uv, material=material)])

    np.testing.assert_allclose(only(read(doc)).uvs, [[0, 0], [1, 0], [0, 32768 / 65535]], rtol=1e-6)


def test_sparse_accessor():
    group = k3d.gltf(os.path.join(ASSETS, "SimpleSparseAccessor.gltf"))
    vertices = only(group).vertices

    # the sparse block moves three of the fourteen vertices up
    assert vertices.shape == (14, 3)
    np.testing.assert_array_equal(vertices[[8, 10, 12], 1], [2, 3, 4])
    assert np.count_nonzero(vertices[:, 1] > 1.5) == 3


def test_a_primitive_without_indices_draws_its_vertices_in_order():
    doc = Document()
    doc.mesh([doc.triangle()])

    np.testing.assert_array_equal(only(read(doc)).indices, [[0, 1, 2]])


@pytest.mark.parametrize("mode, expected", [
    (5, [[0, 1, 2], [1, 3, 2]]),  # strip: every other triangle keeps the winding
    (6, [[1, 2, 0], [2, 3, 0]]),  # fan
])
def test_strips_and_fans_become_triangles(mode, expected):
    doc = Document()
    p = doc.accessor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    doc.mesh([{"attributes": {"POSITION": p}, "mode": mode}])

    np.testing.assert_array_equal(only(read(doc)).indices, expected)


def test_points_and_lines_become_points_and_lines():
    group = k3d.gltf(os.path.join(ASSETS, "MeshPrimitiveModes.gltf"))
    kinds = sorted(type(o).__name__ for o in group)

    assert kinds == ["Lines", "Lines", "Lines", "Mesh", "Mesh", "Mesh", "Points"]

    for obj in group:
        if isinstance(obj, Lines):
            assert obj.indices_type == "segment"
            assert obj.indices.shape[1] == 2
        if isinstance(obj, Points):
            # a world size, the same in every renderer
            assert obj.shader == "3d"
            assert obj.point_size > 0


def test_line_loop_closes():
    doc = Document()
    p = doc.accessor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    doc.mesh([{"attributes": {"POSITION": p}, "mode": 2}])

    np.testing.assert_array_equal(only(read(doc)).indices, [[0, 1], [1, 2], [2, 0]])


def test_default_morph_weights_are_applied():
    doc = Document()
    target = doc.accessor([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    doc.json["meshes"].append({"primitives": [doc.triangle(targets=[{"POSITION": target}])],
                               "weights": [0.5]})
    doc.node({"mesh": 0})

    with pytest.warns(UserWarning, match="morph targets"):
        mesh = only(k3d.glb(doc.glb()))

    np.testing.assert_allclose(mesh.vertices[:, 2], [0.5, 0.5, 0.5])


# -- scene graph --------------------------------------------------------------------------

def test_node_hierarchy_composes_into_model_matrices():
    doc = Document()
    child = doc.mesh([doc.triangle()], name="child", node={"translation": [1, 2, 3]})
    doc.json["scenes"][0]["nodes"].remove(child)
    doc.node({"name": "parent", "scale": [2, 2, 2], "children": [child]})

    mesh = only(read(doc, up="z"))
    expected = np.identity(4)
    expected[:3, :3] *= 2
    expected[:3, 3] = [2, 4, 6]

    assert mesh.name == "child"
    np.testing.assert_allclose(mesh.model_matrix, expected)


def test_matrix_nodes_are_column_major():
    doc = Document()
    matrix = np.identity(4)
    matrix[:3, 3] = [5, 6, 7]
    doc.mesh([doc.triangle()], node={"matrix": matrix.T.reshape(-1).tolist()})

    np.testing.assert_allclose(only(read(doc, up="z")).model_matrix, matrix)


def test_rotation_quaternion():
    doc = Document()
    half = np.sqrt(0.5)
    doc.mesh([doc.triangle()], node={"rotation": [0, 0, half, half]})  # 90 degrees about z

    m = only(read(doc, up="z")).model_matrix

    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-6)


def test_y_up_is_turned_to_z_up_by_default():
    doc = Document()
    doc.mesh([doc.triangle()])

    turned = only(read(doc)).model_matrix
    kept = only(read(doc, up="z")).model_matrix

    np.testing.assert_allclose(turned[:3, :3] @ [0, 1, 0], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(kept, np.identity(4))


def test_transform_arguments_place_the_whole_model():
    doc = Document()
    doc.mesh([doc.triangle()], node={"translation": [1, 0, 0]})

    mesh = only(read(doc, up="z", translation=[0, 0, 10]))

    np.testing.assert_allclose(mesh.model_matrix[:3, 3], [1, 0, 10])

    with pytest.raises(TypeError, match="colour"):
        read(doc, colour=3)


def test_group_model_matrix_moves_the_hierarchy_not_each_member():
    doc = Document()
    doc.mesh([doc.triangle()], name="a", node={"translation": [1, 0, 0]})
    doc.mesh([doc.triangle()], name="b", node={"translation": [0, 1, 0]})
    group = read(doc, up="z")
    shift = np.identity(4)
    shift[:3, 3] = [0, 0, 5]

    group.model_matrix = shift

    a, b = group
    np.testing.assert_allclose(a.model_matrix[:3, 3], [1, 0, 5])
    np.testing.assert_allclose(b.model_matrix[:3, 3], [0, 1, 5])
    assert group.transform is not None


def test_scene_by_index_and_by_name():
    doc = Document()
    doc.mesh([doc.triangle()], name="first")
    second = doc.mesh([doc.triangle()], name="second")
    doc.json["scenes"][0]["nodes"].remove(second)
    doc.json["scenes"].append({"name": "other", "nodes": [second]})

    assert only(read(doc)).name == "first"
    assert only(read(doc, scene=1)).name == "second"
    assert only(read(doc, scene="other")).name == "second"

    with pytest.raises(ValueError, match="no scene"):
        read(doc, scene="missing")


def test_instances_become_objects():
    group = k3d.glb(os.path.join(ASSETS, "SimpleInstancing.glb"))

    assert len(group) == 125
    assert len({tuple(np.asarray(o.model_matrix)[:3, 3].round(4)) for o in group}) == 125


def test_negative_scale_reaches_the_model_matrix():
    group = k3d.glb(os.path.join(ASSETS, "NegativeScaleTest.glb"))
    determinants = [np.linalg.det(np.asarray(o.model_matrix)[:3, :3]) for o in group]

    assert min(determinants) < 0 < max(determinants)


# -- materials ----------------------------------------------------------------------------

def test_factors_are_converted_from_linear():
    doc = Document()
    material = doc.material(
        pbrMetallicRoughness={"baseColorFactor": [0.2158605, 1.0, 0.0, 0.25],
                              "metallicFactor": 0.3, "roughnessFactor": 0.6},
        emissiveFactor=[1.0, 0.0, 0.2158605],
        extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
        doubleSided=True,
    )
    doc.mesh([doc.triangle(material=material)])

    mesh = only(read(doc))

    # linear 0.2159 is sRGB 128/255 - the value K3D displays
    assert mesh.color == 0x80FF00
    assert mesh.opacity == pytest.approx(0.25)
    assert mesh.metalness == pytest.approx(0.3)
    assert mesh.roughness == pytest.approx(0.6)
    assert mesh.emissive == 0xFF0080
    assert mesh.emissive_intensity == pytest.approx(4.0)
    assert mesh.side == "double"


def test_the_default_material_is_glTFs():
    doc = Document()
    doc.mesh([doc.triangle()])

    mesh = only(read(doc))

    assert (mesh.color, mesh.metalness, mesh.roughness) == (0xFFFFFF, 1.0, 1.0)
    assert (mesh.alpha_mode, mesh.side, mesh.emissive) == ("opaque", "front", 0)


@pytest.mark.parametrize("mode, cutoff, expected", [
    ("BLEND", None, ("blend", 0.5)),
    ("MASK", 0.3, ("mask", 0.3)),
    ("OPAQUE", None, ("opaque", 0.5)),
])
def test_alpha_modes(mode, cutoff, expected):
    doc = Document()
    definition = {"alphaMode": mode}
    if cutoff is not None:
        definition["alphaCutoff"] = cutoff
    doc.mesh([doc.triangle(material=doc.material(**definition))])

    mesh = only(read(doc))

    assert (mesh.alpha_mode, pytest.approx(mesh.alpha_cutoff)) == (expected[0], expected[1])


def test_every_texture_slot_and_its_strengths():
    doc = Document()
    t = [doc.image(PNG), doc.image(JPEG, "image/jpeg"), doc.image(PNG), doc.image(PNG), doc.image(PNG)]
    material = doc.material(
        pbrMetallicRoughness={"baseColorTexture": {"index": t[0]},
                              "metallicRoughnessTexture": {"index": t[1]}},
        normalTexture={"index": t[2], "scale": 0.5},
        occlusionTexture={"index": t[3], "strength": 0.25, "texCoord": 1},
        emissiveTexture={"index": t[4]},
    )
    uv0 = doc.accessor([[0, 0], [1, 0], [0, 1]])
    uv1 = doc.accessor([[0.5, 0.5], [1, 0.5], [0.5, 1]])
    doc.mesh([doc.triangle(TEXCOORD_0=uv0, TEXCOORD_1=uv1, material=material)])

    mesh = only(read(doc))

    assert mesh.texture == PNG and mesh.texture_file_format == "png"
    assert mesh.metalness_roughness_map == JPEG
    assert mesh.normal_map == PNG and mesh.normal_scale == pytest.approx(0.5)
    assert mesh.occlusion_map == PNG and mesh.occlusion_strength == pytest.approx(0.25)
    assert mesh.emissive_map == PNG
    np.testing.assert_array_equal(mesh.uvs, [[0, 0], [1, 0], [0, 1]])
    # occlusion reads the second set
    np.testing.assert_array_equal(mesh.uvs2, [[0.5, 0.5], [1, 0.5], [0.5, 1]])


def test_occlusion_on_the_main_set_needs_no_second_one():
    doc = Document()
    t = doc.image(PNG)
    material = doc.material(pbrMetallicRoughness={"baseColorTexture": {"index": t}},
                            occlusionTexture={"index": t})
    doc.mesh([doc.triangle(TEXCOORD_0=doc.accessor([[0, 0], [1, 0], [0, 1]]), material=material)])

    assert only(read(doc)).uvs2.size == 0


def test_texture_transform_is_baked_into_the_uvs():
    doc = Document()
    t = doc.image(PNG)
    transform = {"offset": [0.5, 0.0], "scale": [2.0, 1.0], "rotation": 0.0}
    material = doc.material(pbrMetallicRoughness={"baseColorTexture": {
        "index": t, "extensions": {"KHR_texture_transform": transform}}})
    doc.mesh([doc.triangle(TEXCOORD_0=doc.accessor([[0, 0], [1, 0], [0, 1]]), material=material)])

    np.testing.assert_allclose(only(read(doc)).uvs, [[0.5, 0], [2.5, 0], [0.5, 1]])


def test_sampler_wrapping_per_axis():
    group = k3d.glb(os.path.join(ASSETS, "TextureSettingsTest.glb"))
    wraps = {o.texture_wrap for o in group if o.texture}

    assert {"repeat", "clamp repeat", "repeat clamp", "mirror repeat", "repeat mirror"} <= wraps


def test_vertex_colours_with_alpha():
    doc = Document()
    rgba = doc.accessor(np.array([[1, 0, 0, 1], [0, 1, 0, 0.5], [0, 0, 1, 0]], np.float32))
    material = doc.material(alphaMode="BLEND")
    doc.mesh([doc.triangle(COLOR_0=rgba, material=material)])

    mesh = only(read(doc))

    np.testing.assert_array_equal(mesh.colors, [0xFF0000, 0x00FF00, 0x0000FF])
    np.testing.assert_allclose(mesh.opacities, [1, 0.5, 0])


def test_unlit_emits_its_base_colour():
    doc = Document()
    t = doc.image(PNG)
    material = doc.material(pbrMetallicRoughness={"baseColorFactor": [1, 0, 0, 1],
                                                  "baseColorTexture": {"index": t}},
                            extensions={"KHR_materials_unlit": {}})
    doc.json["extensionsRequired"] = ["KHR_materials_unlit"]
    doc.mesh([doc.triangle(TEXCOORD_0=doc.accessor([[0, 0], [1, 0], [0, 1]]), material=material)])

    mesh = only(read(doc))

    assert (mesh.color, mesh.emissive, mesh.emissive_map) == (0, 0xFF0000, PNG)


def test_what_is_left_out_is_said_once():
    doc = Document()
    doc.mesh([doc.triangle(material=doc.material(extensions={"KHR_materials_clearcoat": {}}))])
    doc.json["animations"] = [{"channels": [], "samplers": []}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        k3d.glb(doc.glb())

    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "KHR_materials_clearcoat" in messages[0] and "animations" in messages[0]


# -- the group ----------------------------------------------------------------------------

def test_members_share_the_group_and_carry_names():
    group = k3d.glb(os.path.join(ASSETS, "VertexColorTest.glb"))

    assert isinstance(group, Group)
    assert {o.group for o in group} == {"VertexColorTest"}
    assert all(o.name for o in group)
    assert all("gltf_node" in o.custom_data for o in group)
    assert group[list(group)[0].name] is list(group)[0]


def test_group_and_visibility_and_compression_reach_every_member():
    doc = Document()
    doc.mesh([doc.triangle(), doc.triangle()], name="pair")

    group = read(doc, group="parts", visible=False, compression_level=7)

    assert [(o.group, o.visible, o.compression_level) for o in group] == [("parts", False, 7)] * 2
    assert sorted(o.name for o in group) == ["pair/0", "pair/1"]


def test_a_read_model_goes_into_a_plot_whole():
    plot = k3d.plot()
    group = k3d.glb(os.path.join(ASSETS, "VertexColorTest.glb"))

    plot += group
    assert len(plot.objects) == len(group)

    plot -= group
    assert len(plot.objects) == 0


def test_every_member_is_an_ordinary_object():
    for obj in k3d.glb(os.path.join(ASSETS, "BoxTextured.glb")):
        assert isinstance(obj, Mesh)
        # a snapshot serializes it like any other mesh
        assert list(obj.get_binary()["texture"]["shape"]) == [len(obj.texture)]
