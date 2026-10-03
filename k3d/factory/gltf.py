"""k3d.glb() and k3d.gltf(): glTF 2.0 scenes as K3D objects, one per primitive, read with numpy alone."""

import base64
import json
import os
import struct
import warnings
from typing import Any, Optional, Union
from urllib.parse import unquote

import numpy as np

from ..helpers import image_format
from ..objects import Group, Lines, Mesh, Points
from ..transform import Transform

GLB_MAGIC = b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

# the divisor glTF uses to turn a normalized integer into a float
NORMALIZED = {
    np.dtype(np.int8): 127.0,
    np.dtype(np.uint8): 255.0,
    np.dtype(np.int16): 32767.0,
    np.dtype(np.uint16): 65535.0,
    np.dtype(np.uint32): 4294967295.0,
}

TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

WRAPPING = {10497: "repeat", 33071: "clamp", 33648: "mirror"}

# primitive modes
POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = range(7)

# extensions read in full, or whose absence costs nothing visible
SUPPORTED_EXTENSIONS = {
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
    "KHR_mesh_quantization",
    "KHR_texture_transform",
    "EXT_mesh_gpu_instancing",
    "EXT_texture_webp",
}

# turning glTF's Y up into K3D's Z up: +90 degrees about x
Y_UP_TO_Z_UP = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)

TRANSFORM_ARGUMENTS = ("translation", "rotation", "scaling", "model_matrix", "transform")


def _uv_transform(info):
    """The uv set a textureInfo reads and its KHR_texture_transform as a 3x3 matrix."""
    transform = info.get("extensions", {}).get("KHR_texture_transform")

    if transform is None:
        return info.get("texCoord", 0), None

    ox, oy = transform.get("offset", [0.0, 0.0])
    sx, sy = transform.get("scale", [1.0, 1.0])
    angle = transform.get("rotation", 0.0)
    c, s = np.cos(angle), np.sin(angle)

    translation = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    scale = np.diag([sx, sy, 1.0])

    return transform.get("texCoord", info.get("texCoord", 0)), translation @ rotation @ scale


def _same_uvs(a, b):
    """Whether two (uv set, transform) pairs give the same coordinates."""
    if a[0] != b[0]:
        return False
    ma = a[1] if a[1] is not None else np.identity(3)
    mb = b[1] if b[1] is not None else np.identity(3)
    return np.allclose(ma, mb)


def _srgb(linear):
    """Linear (glTF) to displayed (K3D) colour."""
    linear = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)

    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def _pack(rgb):
    """Packed 0xRRGGBB out of rows of display-space floats in 0..1."""
    channels = np.clip(np.rint(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint32)

    return (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]


def _node_matrix(node):
    """A node's local transform as a row-major 4x4 matrix."""
    if "matrix" in node:
        # column-major in the file
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)

    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

    matrix = np.identity(4)
    matrix[:3, :3] = rotation * s
    matrix[:3, 3] = t

    return matrix


class _Reader:
    """One glTF document: its JSON, its buffers, and decoding of what refers to them."""

    def __init__(self, document, binary_chunk, base_dir, label):
        self.document = document
        self.binary_chunk = binary_chunk
        self.base_dir = base_dir
        self.label = label
        self.ignored = set()
        self._buffers = {}
        self._images = {}

        asset = document.get("asset", {})
        version = str(asset.get("version", ""))

        if not version.startswith("2."):
            raise ValueError("%s is glTF %s - only glTF 2.0 is read" % (label, version or "of no version"))

        required = set(document.get("extensionsRequired", [])) - SUPPORTED_EXTENSIONS

        if required:
            compressed = required & {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}
            hint = (" - decompress it first, e.g. `npx @gltf-transform/cli copy in.glb out.glb`"
                    if compressed else "")
            raise NotImplementedError(
                "%s requires %s, which K3D does not read%s" % (label, ", ".join(sorted(required)), hint)
            )

    def warn(self, what):
        self.ignored.add(what)

    # -- raw data ---------------------------------------------------------------------------

    def _uri(self, uri, what):
        if uri.startswith("data:"):
            header, _, payload = uri.partition(",")

            if header.endswith(";base64"):
                return base64.b64decode(payload)

            return unquote(payload).encode("latin-1")

        if self.base_dir is None:
            raise ValueError(
                "%s refers to the external file %r for %s - pass the path of the .gltf file "
                "rather than its bytes, so the file can be found next to it" % (self.label, uri, what)
            )

        with open(os.path.join(self.base_dir, unquote(uri)), "rb") as f:
            return f.read()

    def buffer(self, index):
        if index not in self._buffers:
            definition = self.document["buffers"][index]

            if "uri" in definition:
                data = self._uri(definition["uri"], "buffer %d" % index)
            elif index == 0 and self.binary_chunk is not None:
                data = self.binary_chunk
            else:
                raise ValueError("%s: buffer %d has no data" % (self.label, index))

            self._buffers[index] = memoryview(data)

        return self._buffers[index]

    def buffer_view(self, index):
        view = self.document["bufferViews"][index]
        offset = view.get("byteOffset", 0)

        return self.buffer(view["buffer"])[offset:offset + view["byteLength"]], view.get("byteStride")

    def accessor(self, index):
        """The accessor as a float32 or integer array of shape (count, components)."""
        accessor = self.document["accessors"][index]
        dtype = np.dtype(COMPONENT_TYPES[accessor["componentType"]])
        components = TYPE_SIZES[accessor["type"]]
        count = accessor["count"]

        if "bufferView" in accessor:
            data, stride = self.buffer_view(accessor["bufferView"])
            offset = accessor.get("byteOffset", 0)
            element = dtype.itemsize * components

            if stride and stride != element:
                # interleaved: a strided view over the shared buffer, then a compact copy
                layout = np.dtype({"names": ["v"], "formats": [(dtype, (components,))],
                                   "offsets": [0], "itemsize": stride})
                # numpy wants a whole stride after the last element too
                raw = bytes(data[offset:offset + count * stride])
                raw += bytes(1) * (count * stride - len(raw))
                values = np.frombuffer(raw, dtype=layout, count=count)["v"].copy()
            else:
                values = np.frombuffer(data, dtype=dtype, count=count * components, offset=offset).copy()

            values = values.reshape(count, components)
        else:
            # no buffer view: zeros, which a sparse accessor may then fill in
            values = np.zeros((count, components), dtype=dtype)

        if "sparse" in accessor:
            sparse = accessor["sparse"]
            idx_view, _ = self.buffer_view(sparse["indices"]["bufferView"])
            idx_dtype = np.dtype(COMPONENT_TYPES[sparse["indices"]["componentType"]])
            positions = np.frombuffer(idx_view, dtype=idx_dtype, count=sparse["count"],
                                      offset=sparse["indices"].get("byteOffset", 0))
            val_view, _ = self.buffer_view(sparse["values"]["bufferView"])
            replacement = np.frombuffer(val_view, dtype=dtype, count=sparse["count"] * components,
                                        offset=sparse["values"].get("byteOffset", 0))
            values[positions.astype(np.int64)] = replacement.reshape(-1, components)

        if accessor.get("normalized", False) and dtype in NORMALIZED:
            values = np.maximum(values.astype(np.float32) / NORMALIZED[dtype], -1.0)
        elif dtype == np.float32:
            values = values.astype(np.float32, copy=False)

        return values

    def image(self, index):
        """Encoded bytes of an image, or None when no browser can be relied on to decode it."""
        if index in self._images:
            return self._images[index]

        definition = self.document["images"][index]

        if "bufferView" in definition:
            data, _ = self.buffer_view(definition["bufferView"])
            data = bytes(data)
        elif "uri" in definition:
            data = self._uri(definition["uri"], "image %d" % index)
        else:
            data = None

        if data is not None and image_format(data) not in ("png", "jpeg", "webp", "gif"):
            self.warn("images other than PNG, JPEG and WebP (KTX2/Basis among them)")
            data = None

        self._images[index] = data

        return data

    def texture(self, info):
        """Image bytes and the sampler of a textureInfo, or (None, None)."""
        if info is None:
            return None, None

        texture = self.document["textures"][info["index"]]
        extensions = texture.get("extensions", {})
        source = texture.get("source")

        if "EXT_texture_webp" in extensions:
            source = extensions["EXT_texture_webp"]["source"]
        elif source is None:
            self.warn("textures in compressed formats (KHR_texture_basisu)")
            return None, None

        sampler = None
        if "sampler" in texture:
            sampler = self.document["samplers"][texture["sampler"]]

        return self.image(source), sampler


def _read_source(source):
    """The bytes of a source, and the directory relative references resolve against."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None, None

    if hasattr(source, "read"):
        return source.read(), None, None

    path = os.fspath(source)

    with open(path, "rb") as f:
        return f.read(), os.path.dirname(os.path.abspath(path)), path


def _parse_glb(data, label):
    magic, version, length = struct.unpack_from("<4sII", data, 0)

    if magic != GLB_MAGIC:
        raise ValueError("%s is not a binary glTF (.glb) - for a .gltf file use k3d.gltf()" % label)
    if version != 2:
        raise ValueError("%s is GLB version %d - only version 2 is read" % (label, version))

    document = None
    binary_chunk = None
    offset = 12

    while offset + 8 <= min(length, len(data)):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length

        if chunk_type == CHUNK_JSON:
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == CHUNK_BIN and binary_chunk is None:
            binary_chunk = chunk

    if document is None:
        raise ValueError("%s has no JSON chunk" % label)

    return document, binary_chunk


def _parse_gltf(data, label):
    if data[:4] == GLB_MAGIC:
        raise ValueError("%s is a binary glTF (.glb) - use k3d.glb()" % label)

    try:
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        raise ValueError("%s is not glTF JSON" % label) from None


def _strip_indices(count):
    i = np.arange(max(count - 2, 0))
    even = (i % 2) == 0
    a = i
    b = np.where(even, i + 1, i + 2)
    c = np.where(even, i + 2, i + 1)

    return np.stack([a, b, c], axis=1)


def _triangles(mode, indices):
    if mode == TRIANGLES:
        return indices.reshape(-1, 3)
    if mode == TRIANGLE_STRIP:
        return indices[_strip_indices(len(indices))]
    # fan: every triangle shares the first vertex
    i = np.arange(1, max(len(indices) - 1, 1))
    return np.stack([indices[i], indices[i + 1], np.full(len(i), indices[0])], axis=1)


def _segments(mode, indices):
    if mode == LINES:
        return indices.reshape(-1, 2)
    pairs = np.stack([indices[:-1], indices[1:]], axis=1)
    if mode == LINE_LOOP and len(indices) > 1:
        pairs = np.vstack([pairs, [[indices[-1], indices[0]]]])
    return pairs


class _Builder:
    """Turns the primitives of a document into K3D objects under a hierarchy of Transforms."""

    def __init__(self, reader, group, compression_level, visible):
        self.reader = reader
        self.group = group
        self.compression_level = compression_level
        self.visible = visible
        self.objects = []
        self.document = reader.document

    def material(self, index):
        """K3D mesh parameters of a glTF material."""
        reader = self.reader
        material = self.document["materials"][index] if index is not None else {}
        pbr = material.get("pbrMetallicRoughness", {})
        extensions = material.get("extensions", {})

        for name in extensions:
            if name not in ("KHR_materials_emissive_strength", "KHR_materials_unlit"):
                reader.warn(name)

        if "KHR_materials_pbrSpecularGlossiness" in extensions and "pbrMetallicRoughness" not in material:
            # the archived workflow: its diffuse is the closest thing to a base colour
            spec_gloss = extensions["KHR_materials_pbrSpecularGlossiness"]
            pbr = {
                "baseColorFactor": spec_gloss.get("diffuseFactor", [1, 1, 1, 1]),
                "baseColorTexture": spec_gloss.get("diffuseTexture"),
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0 - spec_gloss.get("glossinessFactor", 1.0),
            }

        factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        emissive = material.get("emissiveFactor", [0.0, 0.0, 0.0])
        strength = extensions.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)

        params = {
            "color": int(_pack(_srgb(factor[:3]))),
            "opacity": float(factor[3]),
            "metalness": float(np.clip(pbr.get("metallicFactor", 1.0), 0, 1)),
            "roughness": float(np.clip(pbr.get("roughnessFactor", 1.0), 0, 1)),
            "side": "double" if material.get("doubleSided", False) else "front",
            "alpha_mode": {"BLEND": "blend", "MASK": "mask"}.get(material.get("alphaMode"), "opaque"),
            "alpha_cutoff": float(material.get("alphaCutoff", 0.5)),
            "emissive": int(_pack(_srgb(emissive))),
            "emissive_intensity": float(strength),
        }

        # the texture slots, and the uv set each of them reads
        slots = {
            "texture": pbr.get("baseColorTexture"),
            "metalness_roughness_map": pbr.get("metallicRoughnessTexture"),
            "normal_map": material.get("normalTexture"),
            "occlusion_map": material.get("occlusionTexture"),
            "emissive_map": material.get("emissiveTexture"),
        }
        uv_sets = {}
        wrap = None

        for trait, info in slots.items():
            image, sampler = reader.texture(info)

            if image is None:
                continue

            params[trait] = image
            uv_sets[trait] = _uv_transform(info)

            if wrap is None or trait == "texture":
                sampler = sampler or {}
                wrap_s = WRAPPING.get(sampler.get("wrapS", 10497), "repeat")
                wrap_t = WRAPPING.get(sampler.get("wrapT", 10497), "repeat")

                wrap = wrap_s if wrap_s == wrap_t else "%s %s" % (wrap_s, wrap_t)

        if "normal_map" in params:
            params["normal_scale"] = float(slots["normal_map"].get("scale", 1.0))
        if "occlusion_map" in params:
            params["occlusion_strength"] = float(slots["occlusion_map"].get("strength", 1.0))
        if wrap is not None:
            params["texture_wrap"] = wrap

        if "KHR_materials_unlit" in extensions:
            # unlit: the base colour emitted, the black diffuse keeping only the texture alpha
            params["emissive"] = params["color"]
            params["emissive_intensity"] = 1.0
            params["color"] = 0
            params["metalness"] = 0.0
            params["roughness"] = 1.0
            if "texture" in params:
                params["emissive_map"] = params["texture"]
                uv_sets["emissive_map"] = uv_sets["texture"]
            params["_unlit"] = True

        return params, uv_sets

    def attributes(self, primitive, weights):
        """The vertex attributes of a primitive, with the default morph weights applied."""
        reader = self.reader
        attributes = {name: reader.accessor(index) for name, index in primitive["attributes"].items()}
        targets = primitive.get("targets", [])

        if targets:
            if weights is not None and any(w != 0 for w in weights):
                for target, weight in zip(targets, weights):
                    for name in ("POSITION", "NORMAL"):
                        if name in target and name in attributes and weight != 0:
                            attributes[name] = attributes[name] + weight * reader.accessor(target[name])

            reader.warn("morph targets (shown with their default weights)")

        return attributes

    def primitive(self, primitive, name, weights, transform, custom_data):
        reader = self.reader
        attributes = self.attributes(primitive, weights)

        if "POSITION" not in attributes:
            return

        vertices = attributes["POSITION"].astype(np.float32)
        count = len(vertices)
        mode = primitive.get("mode", TRIANGLES)

        if "indices" in primitive:
            indices = reader.accessor(primitive["indices"]).reshape(-1).astype(np.uint32)
        else:
            indices = np.arange(count, dtype=np.uint32)

        material_index = primitive.get("material")
        params, uv_sets = self.material(material_index)
        common = {
            "name": name,
            "group": self.group,
            "custom_data": custom_data,
            "compression_level": self.compression_level,
            "visible": self.visible,
        }

        unlit = params.pop("_unlit", False)
        colors = None
        opacities = None
        if "COLOR_0" in attributes and unlit:
            reader.warn("vertex colours of unlit materials")
        if "COLOR_0" in attributes:
            rgba = attributes["COLOR_0"].astype(np.float32)
            colors = _pack(_srgb(rgba[:, :3])).astype(np.uint32)
            if rgba.shape[1] == 4:
                opacities = rgba[:, 3].astype(np.float32)

        if mode in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
            obj = self.mesh(vertices, _triangles(mode, indices), attributes, params, uv_sets,
                            colors, opacities, common)
        elif mode == POINTS:
            # sized by _size_points in world units, which every renderer, cinematic too, agrees on
            obj = Points(
                positions=vertices,
                colors=colors if colors is not None else [],
                color=params["color"],
                opacity=params["opacity"],
                opacities=opacities if opacities is not None else [],
                shader="3d",
                **common,
            )
        else:
            obj = Lines(
                vertices=vertices,
                indices=_segments(mode, indices).astype(np.uint32),
                indices_type="segment",
                colors=colors if colors is not None else [],
                color=params["color"],
                opacity=params["opacity"],
                shader="simple",
                width=0.01,
                **common,
            )

        if obj is None:
            return

        transform.add_drawable(obj)
        obj.transform = transform
        obj.model_matrix = transform.model_matrix
        self.objects.append(obj)

    def mesh(self, vertices, triangles, attributes, params, uv_sets, colors, opacities, common):
        reader = self.reader
        uvs = None
        uvs2 = None

        def uv(uv_set):
            index, transform = uv_set
            key = "TEXCOORD_%d" % index
            if key not in attributes:
                return None
            coordinates = attributes[key].astype(np.float64)
            if transform is not None:
                # KHR_texture_transform baked into the coordinates
                coordinates = coordinates @ transform[:2, :2].T + transform[:2, 2]
            return coordinates.astype(np.float32)

        # every map but occlusion reads `uvs`, occlusion reads `uvs2` when it is set
        others = [trait for trait in uv_sets if trait != "occlusion_map"]
        main = uv_sets.get("texture") or (uv_sets[others[0]] if others else None)

        if any(not _same_uvs(uv_sets[trait], main) for trait in others):
            reader.warn("textures of one material reading different uv sets or transforms "
                        "(the base colour's are used)")

        if uv_sets:
            occlusion = uv_sets.get("occlusion_map")
            uvs = uv(main if main is not None else occlusion)

            if occlusion is not None and main is not None and not _same_uvs(occlusion, main):
                uvs2 = uv(occlusion)

            if uvs is None:
                reader.warn("textures on primitives without texture coordinates")

        normals = attributes.get("NORMAL")
        alpha_mode = params.pop("alpha_mode")

        return Mesh(
            vertices=vertices,
            indices=triangles.astype(np.uint32),
            normals=normals.astype(np.float32) if normals is not None else [],
            # glTF: a primitive without normals is flat shaded
            flat_shading=normals is None,
            colors=colors if colors is not None else [],
            opacities=opacities if opacities is not None else [],
            uvs=uvs if uvs is not None else [],
            uvs2=uvs2 if uvs2 is not None else [],
            texture_file_format=image_format(params["texture"]) if "texture" in params else None,
            alpha_mode=alpha_mode,
            attribute=[],
            triangles_attribute=[],
            color_map=[],
            color_range=[],
            volume=[],
            volume_bounds=[],
            opacity_function=[],
            slice_planes=[],
            wireframe=False,
            **params,
            **common,
        )

    def node(self, index, parent, path):
        if index in path:
            raise ValueError("%s: node %d is its own ancestor" % (self.reader.label, index))

        node = self.document["nodes"][index]
        transform = Transform(custom_matrix=_node_matrix(node), parent=parent)
        node_name = node.get("name") or "node %d" % index

        if "skin" in node:
            self.reader.warn("skins (shown in the pose the vertices are stored in)")
        if "camera" in node:
            self.reader.warn("cameras")
        if "KHR_lights_punctual" in node.get("extensions", {}):
            self.reader.warn("lights (KHR_lights_punctual)")

        if "mesh" in node:
            self.mesh_node(node, index, node_name, transform)

        for child in node.get("children", []):
            self.node(child, transform, path | {index})

    def mesh_node(self, node, index, node_name, transform):
        mesh = self.document["meshes"][node["mesh"]]
        weights = node.get("weights", mesh.get("weights"))
        primitives = mesh.get("primitives", [])
        instancing = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        instances = [transform]

        if instancing is not None:
            instances = [Transform(custom_matrix=m, parent=transform)
                         for m in self.instance_matrices(instancing["attributes"])]

        for i, instance in enumerate(instances):
            for p, primitive in enumerate(primitives):
                name = node_name
                if len(primitives) > 1:
                    material = primitive.get("material")
                    label = (self.document["materials"][material].get("name")
                             if material is not None else None)
                    name = "%s/%s" % (node_name, label or p)
                if len(instances) > 1:
                    name = "%s [%d]" % (name, i)

                custom_data = {"gltf_node": index, "gltf_mesh": node["mesh"], "gltf_primitive": p}
                self.primitive(primitive, name, weights, instance, custom_data)

    def instance_matrices(self, attributes):
        reader = self.reader
        t = reader.accessor(attributes["TRANSLATION"]) if "TRANSLATION" in attributes else None
        r = reader.accessor(attributes["ROTATION"]) if "ROTATION" in attributes else None
        s = reader.accessor(attributes["SCALE"]) if "SCALE" in attributes else None
        count = len(next(a for a in (t, r, s) if a is not None))

        return [
            _node_matrix({
                "translation": t[i].tolist() if t is not None else [0, 0, 0],
                "rotation": r[i].tolist() if r is not None else [0, 0, 0, 1],
                "scale": s[i].tolist() if s is not None else [1, 1, 1],
            })
            for i in range(count)
        ]


def _scene_roots(document, scene, label):
    scenes = document.get("scenes", [])

    if not scenes:
        # no scene: every node nobody claims as a child is a root
        children = {c for node in document.get("nodes", []) for c in node.get("children", [])}
        return [i for i in range(len(document.get("nodes", []))) if i not in children]

    if scene is None:
        scene = document.get("scene", 0)
    elif isinstance(scene, str):
        names = [s.get("name") for s in scenes]
        if scene not in names:
            raise ValueError("%s has no scene %r - it has %s" % (label, scene, names))
        scene = names.index(scene)

    if not 0 <= scene < len(scenes):
        raise ValueError("%s has no scene %d - it has %d" % (label, scene, len(scenes)))

    return scenes[scene].get("nodes", [])


def _root_transform(up, kwargs):
    unknown = set(kwargs) - set(TRANSFORM_ARGUMENTS)
    if unknown:
        raise TypeError("unexpected keyword argument(s): %s" % ", ".join(sorted(unknown)))

    if up not in ("y", "z"):
        raise ValueError("up is the axis the file has pointing up, 'y' (glTF's own) or 'z', not %r" % (up,))

    if "transform" in kwargs:
        root = kwargs["transform"]
        if not isinstance(root, Transform):
            raise ValueError("Provided transform argument is not a Transform object")
    else:
        root = Transform(
            translation=kwargs.get("translation"),
            rotation=kwargs.get("rotation"),
            scaling=kwargs.get("scaling"),
            custom_matrix=kwargs.get("model_matrix"),
        )

    axes = Transform(custom_matrix=Y_UP_TO_Z_UP if up == "y" else np.identity(4), parent=root)

    return root, axes


def _size_points(objects):
    """Points get a hundredth of the model's diagonal, in their own node's units."""
    points = [obj for obj in objects if isinstance(obj, Points)]

    if not points:
        return

    boxes = np.array([obj.get_bounding_box() for obj in objects], dtype=np.float64)
    low, high = boxes[:, 0::2].min(axis=0), boxes[:, 1::2].max(axis=0)
    diagonal = float(np.linalg.norm(high - low)) or 1.0

    for obj in points:
        scale = abs(np.linalg.det(np.asarray(obj.model_matrix, np.float64)[:3, :3])) ** (1.0 / 3.0)
        obj.point_size = max(0.01 * diagonal / (scale or 1.0), 1e-6)


def _load(document, binary_chunk, base_dir, label, scene, up, group, compression_level, visible, kwargs):
    reader = _Reader(document, binary_chunk, base_dir, label)
    root, axes = _root_transform(up, kwargs)
    builder = _Builder(reader, group, compression_level, visible)

    for index in _scene_roots(document, scene, label):
        builder.node(index, axes, set())

    _size_points(builder.objects)

    if document.get("animations"):
        reader.warn("animations")

    if reader.ignored:
        warnings.warn("%s: K3D does not show %s" % (label, "; ".join(sorted(reader.ignored))),
                      stacklevel=3)

    if not builder.objects:
        warnings.warn("%s holds no geometry in the scene that was read" % label, stacklevel=3)

    return Group(builder.objects, transform=root)


def _label(path, data_label):
    return os.path.basename(path) if path else data_label


def _group_name(group, path, data_label):
    if group is not None:
        return group
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return data_label


Source = Union[str, os.PathLike, bytes, bytearray, memoryview, Any]


def glb(
        source: Source,
        scene: Optional[Union[int, str]] = None,
        up: str = "y",
        group: Optional[str] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> Group:
    """Read a binary glTF (.glb) scene as K3D objects.

    Every primitive of every mesh in the scene becomes its own :func:`k3d.mesh` - or
    :func:`k3d.points` / :func:`k3d.lines` for point and line primitives - with the
    node's name, and all of them share `group`, so the panel shows the model as one
    folder of parts. They come back together as a :class:`k3d.objects.Group`:
    ``plot += k3d.glb('model.glb')`` adds the whole model.

    Materials map onto mesh parameters: base colour (`color`, `opacity`, `texture`),
    metallic-roughness, normal, occlusion and emissive maps, `alpha_mode` and
    `alpha_cutoff`, double-sidedness and the texture wrapping. Factors and vertex
    colours, which glTF stores as linear values, are converted to the display values
    K3D colours are. Node transforms are kept as a hierarchy of
    :class:`k3d.transform.Transform`: each object's `model_matrix` is its node's, and
    the group's `model_matrix` (or `transform`) moves the whole model.

    Not shown, with one warning naming what was left out: animations, skins (the
    vertices are shown as stored), cameras and lights, KHR_texture_transform, Draco and
    meshopt compression, KTX2 textures, and material extensions other than emissive
    strength. Morph targets are applied with their default weights.

    .. versionadded:: 3.2.0

    Parameters
    ----------
    source : str, os.PathLike, bytes or file-like
        Path of a .glb file, or its bytes - a .glb is self-contained, so bytes from
        anywhere do, e.g. ``plot.fetch_gltf()`` output.
    scene : int or str, optional
        Index or name of the scene to read, by default the file's default scene.
    up : {'y', 'z'}, optional
        The axis pointing up in the file. glTF is 'y' up and K3D 'z' up, so by default
        the model is turned upright; 'z' keeps the coordinates as they are - K3D's own
        exports are z up.
    group : str, optional
        Group of every object in the panel, by default the file name without extension.
    compression_level : int, optional
        Level of data compression [-1, 9] of every object, by default 0.
    visible : bool, optional
        Whether the objects are drawn, by default True.
    **kwargs
        translation, rotation, scaling, model_matrix, or transform - placing the whole
        model, see :ref:`process_transform_arguments`.

    Returns
    -------
    Group
        The objects of the scene.
    """
    data, base_dir, path = _read_source(source)
    label = _label(path, "GLB data")
    document, binary_chunk = _parse_glb(data, label)

    return _load(document, binary_chunk, base_dir, label, scene, up,
                 _group_name(group, path, "glb"), compression_level, visible, kwargs)


def gltf(
        source: Source,
        scene: Optional[Union[int, str]] = None,
        up: str = "y",
        group: Optional[str] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> Group:
    """Read a glTF (.gltf) scene as K3D objects.

    The JSON form of glTF: buffers and images are files next to it or data: URIs inside
    it. Everything else - what becomes which object, what is converted, what is left
    out - is as in :func:`k3d.glb`, which reads the binary form.

    .. versionadded:: 3.2.0

    Parameters
    ----------
    source : str, os.PathLike, bytes or file-like
        Path of a .gltf file. Bytes are read too, as long as every buffer and image is
        inlined as a data: URI - there is no directory to find the other files in.
    scene : int or str, optional
        Index or name of the scene to read, by default the file's default scene.
    up : {'y', 'z'}, optional
        The axis pointing up in the file, by default 'y' as glTF has it; see :func:`k3d.glb`.
    group : str, optional
        Group of every object in the panel, by default the file name without extension.
    compression_level : int, optional
        Level of data compression [-1, 9] of every object, by default 0.
    visible : bool, optional
        Whether the objects are drawn, by default True.
    **kwargs
        translation, rotation, scaling, model_matrix, or transform - placing the whole
        model, see :ref:`process_transform_arguments`.

    Returns
    -------
    Group
        The objects of the scene.
    """
    data, base_dir, path = _read_source(source)
    label = _label(path, "glTF data")
    document = _parse_gltf(data, label)

    return _load(document, None, base_dir, label, scene, up,
                 _group_name(group, path, "gltf"), compression_level, visible, kwargs)
