// "Everything mesh" (stage 2 of renderer_cinematic.md): a parallel scene of
// plain THREE.Mesh + MeshStandardMaterial mirroring K3DObjects for the path
// tracer. K3DObjects stays the source of truth and is never modified; shapes
// are preserved, implementations replaced - impostors become icospheres,
// screen ribbons become world-width tubes, GPU colormaps get baked CPU-side.
const THREE = require('three');
const { mergeGeometries, mergeVertices } = require('three/examples/jsm/utils/BufferGeometryUtils');
const streamLine = require('../../helpers/Streamline');
const Fn = require('../../helpers/Fn');
const buffer = require('../../../../core/lib/helpers/buffer');
const colorMapHelper = require('../../../../core/lib/helpers/colorMap');

// The rasteriser draws point spheres as instances, so mesh_detail costs it
// almost nothing; the path tracer needs every sphere as real geometry in one
// BVH. This ceiling is where that stops being reasonable - it leaves the
// default mesh_detail intact for tens of thousands of points (66k at detail 2,
// 20k at detail 3) and only degrades, loudly, beyond that.
const TRIANGLE_BUDGET = 12000000;

function hasData(field) {
    return field && field.data && field.data.length > 0;
}

function opacityFunctionOf(json) {
    return hasData(json.opacity_function) ? json.opacity_function.data : null;
}

// color_range is a plain 2-element array on the wire, unlike the ndarrays
function usesColorMap(json) {
    return hasData(json.attribute) && hasData(json.color_map)
        && json.color_range && json.color_range.length === 2;
}

// the same 1024x1 gradient canvas the GPU paths sample, read back once - the
// baked colors match the rasterisers' colormap rendering exactly
function colorMapSampler(colorMap, opacityFunction) {
    const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024, 1, opacityFunction);
    const pixels = canvas.getContext('2d').getImageData(0, 0, 1024, 1).data;

    return function sample(t) {
        const i = Math.max(0, Math.min(1023, Math.round(t * 1023))) * 4;

        return [pixels[i] / 255.0, pixels[i + 1] / 255.0, pixels[i + 2] / 255.0, pixels[i + 3] / 255.0];
    };
}

// LinearFilter-style trilinear sampling of a [z, y, x] scalar grid at
// normalized coordinates - mirrors what the volume shaders read
function trilinearSampler(data, shape) {
    const depth = shape[0];
    const height = shape[1];
    const width = shape[2];

    function at(x, y, z) {
        return data[z * width * height + y * width + x];
    }

    return function sample(u, v, w) {
        const x = Math.min(Math.max(u * width - 0.5, 0), width - 1);
        const y = Math.min(Math.max(v * height - 0.5, 0), height - 1);
        const z = Math.min(Math.max(w * depth - 0.5, 0), depth - 1);
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const z0 = Math.floor(z);
        const x1 = Math.min(x0 + 1, width - 1);
        const y1 = Math.min(y0 + 1, height - 1);
        const z1 = Math.min(z0 + 1, depth - 1);
        const fx = x - x0;
        const fy = y - y0;
        const fz = z - z0;

        const c00 = at(x0, y0, z0) * (1 - fx) + at(x1, y0, z0) * fx;
        const c10 = at(x0, y1, z0) * (1 - fx) + at(x1, y1, z0) * fx;
        const c01 = at(x0, y0, z1) * (1 - fx) + at(x1, y0, z1) * fx;
        const c11 = at(x0, y1, z1) * (1 - fx) + at(x1, y1, z1) * fx;

        return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
    };
}

// the path tracer requires uniform wrap/interpolation flags across every
// texture in the scene - clones get one canonical set
function normalizeTexture(texture) {
    const clone = texture.clone();

    clone.wrapS = THREE.ClampToEdgeWrapping;
    clone.wrapT = THREE.ClampToEdgeWrapping;
    clone.minFilter = THREE.LinearFilter;
    clone.magFilter = THREE.LinearFilter;
    clone.generateMipmaps = false;
    clone.needsUpdate = true;

    return clone;
}

// Standard/Physical pass through (clone drops depth-peel onBeforeCompile and
// expando uniforms by design); MeshBasic is lifted to a rough diffuse Standard.
// Anything else (ShaderMaterial: MeshLine ribbons, slice planes) returns null.
function sanitizeMaterial(material) {
    let clean = null;

    if (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) {
        clean = material.clone();
    } else if (material.isMeshBasicMaterial) {
        clean = new THREE.MeshStandardMaterial({
            color: material.color.clone(),
            vertexColors: material.vertexColors,
            side: material.side,
            opacity: material.opacity,
            roughness: 1.0,
            metalness: 0.0,
        });

        if (material.map) {
            clean.map = material.map;
        }
    } else {
        return null;
    }

    // undo the depth-peel branch flags; the peel pipeline does not exist here
    clean.blending = THREE.NormalBlending;
    clean.transparent = material.transparent || clean.opacity < 1.0;
    clean.depthWrite = !clean.transparent;

    if (clean.map) {
        clean.map = normalizeTexture(clean.map);
    }

    return clean;
}

function bakeWorldMatrix(node, source) {
    node.matrixAutoUpdate = false;
    node.matrix.copy(source.matrixWorld);
    node.matrixWorld.copy(source.matrixWorld);

    return node;
}

// generic passthrough: every visible Mesh leaf whose material survives
// sanitisation, with its world transform baked in
function buildPassthrough(sourceObj) {
    const group = new THREE.Group();

    sourceObj.traverse((obj) => {
        if (!obj.isMesh || !obj.visible || !obj.material || obj.material.color === undefined) {
            return;
        }

        const material = sanitizeMaterial(obj.material);

        if (material === null) {
            return;
        }

        const clone = new THREE.Mesh(obj.geometry, material);

        group.add(bakeWorldMatrix(clone, obj));
    });

    return group.children.length > 0 ? group : null;
}

function pointColors(json, count) {
    if (usesColorMap(json)) {
        const sample = colorMapSampler(json.color_map.data, opacityFunctionOf(json));
        const attribute = json.attribute.data;
        const low = json.color_range[0];
        const span = json.color_range[1] - low || 1.0;
        const colors = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            const rgba = sample((attribute[i] - low) / span);

            colors[i * 3] = rgba[0];
            colors[i * 3 + 1] = rgba[1];
            colors[i * 3 + 2] = rgba[2];
        }

        return colors;
    }

    if (hasData(json.colors) && json.colors.data.length === count) {
        return buffer.colorsToFloat32Array(json.colors.data);
    }

    return null;
}

// merged icospheres: the library traces no instances and no point primitives,
// so N points become one indexed Mesh of N spheres with per-vertex colors
function buildPoints(json) {
    const positions = json.positions.data;
    const count = Math.floor(positions.length / 3);

    if (count === 0) {
        return null;
    }

    const shader = (json.shader || '3d').toLowerCase() === '3dspecular'
        ? '3d' : (json.shader || '3d').toLowerCase();
    const pointSize = typeof json.point_size !== 'undefined' ? json.point_size : 1.0;
    const sizes = (hasData(json.point_sizes) && json.point_sizes.data.length === count)
        ? json.point_sizes.data : null;

    // mesh_detail is the user's own subdivision level - the same trait the
    // rasterised shader='mesh' variant builds its icosahedron from - so it is
    // honoured here too, for every shader: in cinematic even a 'dot' point is a
    // real sphere, and its tessellation is the user's call. The triangle budget
    // only ever lowers it, and says so.
    const requested = typeof json.mesh_detail !== 'undefined' ? json.mesh_detail : 2;
    let detail = Math.max(0, Math.min(12, requested));

    // PolyhedronGeometry ships as a triangle soup (no index) - weld it back
    // so N points do not carry 6x the vertex data
    function icosphereTemplate(level) {
        return mergeVertices(new THREE.IcosahedronGeometry(1, level));
    }

    let template = icosphereTemplate(detail);

    while (detail > 0 && (count * template.index.count) / 3 > TRIANGLE_BUDGET) {
        detail--;
        template = icosphereTemplate(detail);
    }

    if (detail !== requested) {
        console.warn(`K3D.cinematic: ${count} points at mesh_detail ${requested} exceed the `
            + `${TRIANGLE_BUDGET} triangle budget - rendering them at mesh_detail ${detail}`);
    }

    const tPos = template.attributes.position.array;
    const tIndex = template.index.array;
    const vPerSphere = template.attributes.position.count;
    const iPerSphere = tIndex.length;

    const outPos = new Float32Array(count * vPerSphere * 3);
    const outNorm = new Float32Array(count * vPerSphere * 3);
    const outIndex = new Uint32Array(count * iPerSphere);
    const colors = pointColors(json, count);
    const outColor = colors !== null ? new Float32Array(count * vPerSphere * 3) : null;

    for (let i = 0; i < count; i++) {
        // billboards take per-point sizes as absolute world diameters, the
        // instanced mesh variant takes them as point_size multipliers; 'dot'
        // has no world size at all (pixels) - point_size stands in for it
        const radius = shader === 'mesh'
            ? 0.5 * pointSize * (sizes !== null ? sizes[i] : 1.0)
            : 0.5 * (sizes !== null ? sizes[i] : pointSize);
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        const vOff = i * vPerSphere * 3;

        for (let j = 0; j < vPerSphere; j++) {
            const nx = tPos[j * 3];
            const ny = tPos[j * 3 + 1];
            const nz = tPos[j * 3 + 2];

            outPos[vOff + j * 3] = px + radius * nx;
            outPos[vOff + j * 3 + 1] = py + radius * ny;
            outPos[vOff + j * 3 + 2] = pz + radius * nz;
            // a unit icosphere's positions are its normals
            outNorm[vOff + j * 3] = nx;
            outNorm[vOff + j * 3 + 1] = ny;
            outNorm[vOff + j * 3 + 2] = nz;

            if (outColor !== null) {
                outColor[vOff + j * 3] = colors[i * 3];
                outColor[vOff + j * 3 + 1] = colors[i * 3 + 1];
                outColor[vOff + j * 3 + 2] = colors[i * 3 + 2];
            }
        }

        const iOff = i * iPerSphere;
        const base = i * vPerSphere;

        for (let j = 0; j < iPerSphere; j++) {
            outIndex[iOff + j] = base + tIndex[j];
        }
    }

    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute('position', new THREE.BufferAttribute(outPos, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(outNorm, 3));
    geometry.setIndex(new THREE.BufferAttribute(outIndex, 1));

    if (outColor !== null) {
        geometry.setAttribute('color', new THREE.BufferAttribute(outColor, 3));
    }

    const material = new THREE.MeshStandardMaterial({
        roughness: typeof json.roughness !== 'undefined' ? json.roughness : 0.4,
        metalness: typeof json.metalness !== 'undefined' ? json.metalness : 0.0,
        opacity: typeof json.opacity !== 'undefined' ? json.opacity : 1.0,
        vertexColors: outColor !== null,
    });

    if (outColor === null) {
        material.color = new THREE.Color(json.color !== undefined ? json.color : 0xff00);
    }

    material.transparent = material.opacity < 1.0;
    material.depthWrite = !material.transparent;

    return new THREE.Mesh(geometry, material);
}

function tubeMaterial(json, geometry) {
    const material = new THREE.MeshStandardMaterial({
        emissive: 0,
        roughness: typeof json.roughness !== 'undefined' ? json.roughness : 0.4,
        metalness: typeof json.metalness !== 'undefined' ? json.metalness : 0.0,
        opacity: typeof json.opacity !== 'undefined' ? json.opacity : 1.0,
        side: THREE.DoubleSide,
    });

    if (usesColorMap(json)) {
        // Streamline already wrote the uv attribute - this only sets the map
        Fn.handleColorMap(geometry, json.color_map.data, json.color_range, null, material);
        material.map = normalizeTexture(material.map);
    } else {
        material.setValues({ color: 0xffffff, vertexColors: true });
    }

    material.transparent = material.opacity < 1.0;
    material.depthWrite = !material.transparent;

    return material;
}

// line (singular): one polyline with NaN row separators, which Streamline
// splits natively. 'thick' extrudes its FULL width on screen while Streamline
// takes a radius - hence width/2; 'simple' follows the mesh-shader convention
// (width = radius) per the plan's representation table.
function buildLine(json) {
    const vertices = json.vertices.data;
    const count = Math.floor(vertices.length / 3);

    if (count < 2) {
        return null;
    }

    const width = typeof json.width !== 'undefined' ? json.width : 0.01;
    const radius = (json.shader || 'simple') === 'thick' ? width / 2.0 : width;
    const radialSegments = json.radial_segments || 8;
    const colorMapped = usesColorMap(json);
    const verticesColors = (!colorMapped && hasData(json.colors)
        && json.colors.data.length === count)
        ? buffer.colorsToFloat32Array(json.colors.data) : null;

    const geometry = streamLine(
        vertices,
        colorMapped ? json.attribute.data : null,
        radius,
        radialSegments,
        new THREE.Color(json.color !== undefined ? json.color : 0xff00),
        verticesColors,
        colorMapped ? json.color_range : null,
    );

    return new THREE.Mesh(geometry, tubeMaterial(json, geometry));
}

// lines (plural): topology from indices (segment pairs or triangle edges),
// deduplicated undirected - the same contract LinesSimple/LinesMesh apply
function uniqueEdges(json) {
    const vertices = json.vertices.data;
    const indices = Fn.guardIndices(json.indices.data, vertices, 'lines');
    const jump = json.indices_type === 'segment' ? 2 : 3;
    const vertexCount = Math.floor(vertices.length / 3);
    const seen = new Set();
    const edges = [];

    function addEdge(a, b) {
        const key = Math.min(a, b) * vertexCount + Math.max(a, b);

        if (!seen.has(key)) {
            seen.add(key);
            edges.push(a, b);
        }
    }

    for (let i = 0; i + jump <= indices.length; i += jump) {
        if (jump === 2) {
            addEdge(indices[i], indices[i + 1]);
        } else {
            addEdge(indices[i], indices[i + 1]);
            addEdge(indices[i + 1], indices[i + 2]);
            addEdge(indices[i + 2], indices[i]);
        }
    }

    return edges;
}

function buildLines(json) {
    const vertices = json.vertices.data;
    const edges = uniqueEdges(json);

    if (edges.length === 0) {
        return null;
    }

    const width = typeof json.width !== 'undefined' ? json.width : 0.01;
    const radius = (json.shader || 'simple') === 'thick' ? width / 2.0 : width;
    const radialSegments = json.radial_segments || 8;
    const colorMapped = usesColorMap(json);
    const vertexCount = Math.floor(vertices.length / 3);
    const sourceColors = (!colorMapped && hasData(json.colors)
        && json.colors.data.length === vertexCount)
        ? buffer.colorsToFloat32Array(json.colors.data) : null;
    const color = new THREE.Color(json.color !== undefined ? json.color : 0xff00);
    const geometries = [];

    for (let e = 0; e < edges.length; e += 2) {
        const a = edges[e];
        const b = edges[e + 1];
        const points = [
            vertices[a * 3], vertices[a * 3 + 1], vertices[a * 3 + 2],
            vertices[b * 3], vertices[b * 3 + 1], vertices[b * 3 + 2],
        ];
        const attributes = colorMapped
            ? [json.attribute.data[a], json.attribute.data[b]] : null;
        const verticesColors = sourceColors !== null
            ? [
                sourceColors[a * 3], sourceColors[a * 3 + 1], sourceColors[a * 3 + 2],
                sourceColors[b * 3], sourceColors[b * 3 + 1], sourceColors[b * 3 + 2],
            ] : null;

        geometries.push(streamLine(points, attributes, radius, radialSegments, color,
            verticesColors, colorMapped ? json.color_range : null));
    }

    const geometry = mergeGeometries(geometries);

    return new THREE.Mesh(geometry, tubeMaterial(json, geometry));
}

// vectors / vector_field: shaft endpoints and colors read back from the live
// MeshLine geometry (every input vertex is duplicated twice; a segment owns
// expanded vertices 4k..4k+3), which sidesteps the two objects' different
// grid/scalar/matrix conventions. Heads are already merged cones with normals
// and vertex colors - only the unlit material needs lifting.
function buildVectors(json, sourceObj) {
    const group = new THREE.Group();
    const radius = (typeof json.line_width !== 'undefined' ? json.line_width : 0.01) / 2.0;

    sourceObj.children.forEach((child) => {
        if (!child.isMesh || !child.visible) {
            return;
        }

        if (child.material && child.material.type === 'MeshLineMaterial') {
            const pos = child.geometry.attributes.position;
            const col = child.geometry.attributes.colors;
            const segments = Math.floor(pos.count / 4);
            const geometries = [];

            for (let k = 0; k < segments; k++) {
                const o = 4 * k;
                const t = 4 * k + 2;
                const points = [
                    pos.getX(o), pos.getY(o), pos.getZ(o),
                    pos.getX(t), pos.getY(t), pos.getZ(t),
                ];
                const verticesColors = col
                    ? [
                        col.getX(o), col.getY(o), col.getZ(o),
                        col.getX(t), col.getY(t), col.getZ(t),
                    ] : null;

                geometries.push(streamLine(points, null, radius, 8,
                    new THREE.Color(0xffffff), verticesColors, null));
            }

            if (geometries.length > 0) {
                const mesh = new THREE.Mesh(
                    mergeGeometries(geometries),
                    new THREE.MeshStandardMaterial({
                        roughness: 0.4, metalness: 0.0, vertexColors: col !== undefined,
                    }),
                );

                group.add(mesh);
            }

            return;
        }

        if (child.material && child.material.isMeshBasicMaterial
            && child.geometry.attributes.color) {
            group.add(new THREE.Mesh(child.geometry, new THREE.MeshStandardMaterial({
                roughness: 0.4, metalness: 0.0, vertexColors: true,
            })));
        }
    });

    return group.children.length > 0 ? bakeWorldMatrix(group, sourceObj) : null;
}

// texture_text sprites become camera-facing quads frozen at prepare() time
function buildTextureText(sourceObj, camera) {
    const group = new THREE.Group();

    sourceObj.traverse((obj) => {
        if (!obj.isSprite || !obj.visible || !obj.material || !obj.material.map) {
            return;
        }

        const material = new THREE.MeshStandardMaterial({
            map: normalizeTexture(obj.material.map),
            transparent: true,
            alphaTest: 0.5,
            side: THREE.DoubleSide,
            roughness: 1.0,
            metalness: 0.0,
        });
        const quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
        const worldPosition = obj.getWorldPosition(new THREE.Vector3());

        quad.position.copy(worldPosition);
        quad.scale.copy(obj.scale);
        quad.lookAt(camera.position);
        quad.updateMatrix();
        quad.updateMatrixWorld(true);
        quad.matrixAutoUpdate = false;

        group.add(quad);
    });

    return group.children.length > 0 ? group : null;
}

// GPU-colormapped surfaces whose colors live in ShaderMaterial uniforms or
// onBeforeCompile expandos that Material.clone() silently drops: bake the
// scalar field into per-vertex colors instead (linear across triangles - a
// documented V1 approximation of the per-pixel GPU sampling)
function bakeScalarFieldMesh(sourceMesh, json, field, toFieldCoords) {
    const geometry = sourceMesh.geometry.clone();
    const position = geometry.attributes.position;
    const sample = trilinearSampler(field.data, field.shape);
    const colorMap = colorMapSampler(json.color_map.data, opacityFunctionOf(json));
    const low = json.color_range[0];
    const span = json.color_range[1] - low || 1.0;
    const colors = new Float32Array(position.count * 3);

    for (let i = 0; i < position.count; i++) {
        const uvw = toFieldCoords(position.getX(i), position.getY(i), position.getZ(i));
        const rgba = colorMap((sample(uvw[0], uvw[1], uvw[2]) - low) / span);

        colors[i * 3] = rgba[0];
        colors[i * 3 + 1] = rgba[1];
        colors[i * 3 + 2] = rgba[2];
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: typeof json.roughness !== 'undefined' ? json.roughness : 0.4,
        metalness: typeof json.metalness !== 'undefined' ? json.metalness : 0.0,
        opacity: typeof json.opacity !== 'undefined' ? json.opacity : 1.0,
        side: Fn.getSide(json),
        flatShading: geometry.attributes.normal === undefined,
    });

    material.transparent = material.opacity < 1.0;
    material.depthWrite = !material.transparent;

    return bakeWorldMatrix(new THREE.Mesh(geometry, material), sourceMesh);
}

// texture (data variant): the scalar DataTexture and the colormap live in a
// ShaderMaterial the passthrough cannot lift - bake both into one RGBA map
function buildTextureData(json, sourceMesh) {
    const height = json.attribute.shape[0];
    const width = json.attribute.shape[1];
    const data = json.attribute.data;
    const sample = colorMapSampler(json.color_map.data, opacityFunctionOf(json));
    const low = json.color_range[0];
    const span = json.color_range[1] - low || 1.0;
    const rgba = new Uint8Array(width * height * 4);

    for (let i = 0; i < width * height; i++) {
        const c = sample((data[i] - low) / span);

        rgba[i * 4] = Math.round(c[0] * 255);
        rgba[i * 4 + 1] = Math.round(c[1] * 255);
        rgba[i * 4 + 2] = Math.round(c[2] * 255);
        rgba[i * 4 + 3] = Math.round(c[3] * 255);
    }

    const texture = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat,
        THREE.UnsignedByteType);

    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;

    const material = new THREE.MeshStandardMaterial({
        map: texture,
        side: THREE.DoubleSide,
        roughness: 1.0,
        metalness: 0.0,
        transparent: true,
    });

    return bakeWorldMatrix(new THREE.Mesh(sourceMesh.geometry, material), sourceMesh);
}

function isMeshVolume(json) {
    return hasData(json.volume) && hasData(json.volume_bounds)
        && Array.isArray(json.color_range) && json.color_range.length === 2
        && hasData(json.color_map);
}

function buildProxyForObject(sourceObj, json, camera) {
    const type = json.type;

    if (type === 'Label' || type === 'Text' || type === 'Text2d'
        || type === 'Volume' || type === 'MIP' || type === 'VolumeSlice') {
        // labels are camera-mutated per frame; volumes composite in stage 5
        return null;
    }

    if (type === 'Points') {
        const mesh = buildPoints(json);

        return mesh !== null ? bakeWorldMatrix(mesh, sourceObj) : null;
    }

    if ((type === 'Line' || type === 'Lines') && json.shader !== 'mesh') {
        const mesh = type === 'Line' ? buildLine(json) : buildLines(json);

        return mesh !== null ? bakeWorldMatrix(mesh, sourceObj) : null;
    }

    if (type === 'Vectors' || type === 'VectorField') {
        return buildVectors(json, sourceObj);
    }

    if (type === 'TextureText') {
        return buildTextureText(sourceObj, camera);
    }

    if (type === 'Texture' && usesColorMap(json)) {
        return buildTextureData(json, sourceObj);
    }

    if (type === 'Mesh' && isMeshVolume(json)) {
        const bounds = json.volume_bounds.data;

        return bakeScalarFieldMesh(sourceObj, json, json.volume, (x, y, z) => [
            (x - bounds[0]) / ((bounds[1] - bounds[0]) || 1.0),
            (y - bounds[2]) / ((bounds[3] - bounds[2]) || 1.0),
            (z - bounds[4]) / ((bounds[5] - bounds[4]) || 1.0),
        ]);
    }

    if (type === 'MarchingCubes' && usesColorMap(json)) {
        // marching-cubes geometry lives in the unit cube [0,1] - local
        // coordinates ARE the normalized field coordinates (the -0.5 centering
        // sits in the object matrix, verified empirically on the live object)
        return bakeScalarFieldMesh(sourceObj, json, json.attribute,
            (x, y, z) => [x, y, z]);
    }

    return buildPassthrough(sourceObj);
}

module.exports = function createSceneProxy(K3D) {
    // proxy groups cached per object id; ids are stable across reloads while
    // instances are not (addOrUpdateObject swaps them without OBJECT_REMOVED)
    const cache = new Map();

    K3D.on(K3D.events.OBJECT_CHANGE, (change) => {
        if (change && typeof change.id !== 'undefined') {
            cache.delete(String(change.id));
        }
    });
    K3D.on(K3D.events.OBJECT_REMOVED, (id) => {
        cache.delete(String(id));
    });
    // OBJECT_LOADED carries no payload and reloads mutate objects in place
    // (same instance, new data) - only a full drop is safe here. Per-object
    // reuse still serves the common cases: camera moves and GUI edits.
    K3D.on(K3D.events.OBJECT_LOADED, () => {
        cache.clear();
    });

    return {
        // mirrors every visible K3DObjects child into `scene`; returns the
        // number of proxied objects so callers can report empty scenes
        populate(scene, camera) {
            const world = K3D.getWorld();
            const alive = new Set();
            let proxied = 0;

            // in cinematic the rasterising loop never runs, so freshly added
            // objects still carry identity matrixWorld - bake from fresh state
            world.K3DObjects.updateMatrixWorld(true);

            world.K3DObjects.children.forEach((sourceObj) => {
                if (!sourceObj.visible) {
                    return;
                }

                const id = String(sourceObj.K3DIdentifier);
                const json = world.ObjectsListJson[sourceObj.K3DIdentifier];

                if (!json) {
                    return;
                }

                alive.add(id);

                let entry = cache.get(id);

                // camera-frozen billboards cannot be reused between frames
                if (!entry || entry.source !== sourceObj || json.type === 'TextureText') {
                    entry = {
                        source: sourceObj,
                        proxy: buildProxyForObject(sourceObj, json, camera),
                    };
                    cache.set(id, entry);
                }

                if (entry.proxy !== null) {
                    scene.add(entry.proxy);
                    proxied++;
                }
            });

            // objects removed without an event (disable(), reload visible=false)
            Array.from(cache.keys()).forEach((id) => {
                if (!alive.has(id)) {
                    cache.delete(id);
                }
            });

            return proxied;
        },

        invalidate() {
            cache.clear();
        },
    };
};
