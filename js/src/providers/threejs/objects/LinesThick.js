const THREE = require('three');
const { colorsToFloat32Array } = require('../../../core/lib/helpers/buffer');
const MeshLine = require('../helpers/THREE.MeshLine')(THREE);
const Fn = require('../helpers/Fn');

const { commonUpdate } = Fn;
const { areAllChangesResolve } = Fn;
const colorMapHelper = require('../../../core/lib/helpers/colorMap');

const { getColorsArray } = Fn;

/**
 * Loader strategy to handle Lines object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
function create(config, K3D) {
    config.width = typeof (config.width) !== 'undefined' ? config.width : 0.1;

    const material = new MeshLine.MeshLineMaterial({
        color: new THREE.Color(1, 1, 1),
        opacity: config.opacity,
        sizeAttenuation: true,
        transparent: true,
        lineWidth: config.width,
        resolution: new THREE.Vector2(K3D.getWorld().width, K3D.getWorld().height),
        side: THREE.DoubleSide,
    });
    let verticesColors = (config.colors && config.colors.data) || null;
    const color = new THREE.Color(config.color);
    let uvs = null;
    const colorRange = config.color_range;
    const colorMap = (config.color_map && config.color_map.data) || null;
    const attr = (config.attribute && config.attribute.data) || null;
    const modelMatrix = new THREE.Matrix4();
    const vertices = config.vertices.data;
    const indices = config.indices.data;
    const edges = new Set();
    const jump = config.indices_type === 'segment' ? 2 : 3;
    let offsets;

    let positions = [];
    let attribute = [];
    let colors = [];

    const verticesCount = vertices.length / 3;

    verticesColors = (verticesColors && verticesColors.length === vertices.length / 3
        ? colorsToFloat32Array(verticesColors) : getColorsArray(color, vertices.length / 3)
    );

    for (let i = 0; i < indices.length; i += jump) {
        if (jump === 3) {
            offsets = [
                [indices[i], indices[i + 1]],
                [indices[i + 1], indices[i + 2]],
                [indices[i + 2], indices[i]],
            ];
        } else {
            offsets = [
                [indices[i], indices[i + 1]],
            ];
        }

        for (let j = 0; j < offsets.length; j++) {
            const hash = offsets[j][0] > offsets[j][1]
                ? offsets[j][0] + offsets[j][1] * verticesCount
                : offsets[j][1] + offsets[j][0] * verticesCount;

            if (!edges.has(hash)) {
                edges.add(hash);

                const o1 = offsets[j][0] * 3;
                const o2 = offsets[j][1] * 3;

                positions.push(
                    vertices[o1],
                    vertices[o1 + 1],
                    vertices[o1 + 2],
                    vertices[o2],
                    vertices[o2 + 1],
                    vertices[o2 + 2],
                );

                if (verticesColors && verticesColors.length > 0) {
                    colors.push(
                        verticesColors[o1],
                        verticesColors[o1 + 1],
                        verticesColors[o1 + 2],
                        verticesColors[o2],
                        verticesColors[o2 + 1],
                        verticesColors[o2 + 2],
                    );
                }

                if (attr && attr.length > 0) {
                    attribute.push(attr[offsets[j][0]], attr[offsets[j][1]]);
                }
            }
        }
    }

    positions = new Float32Array(positions);
    attribute = new Float32Array(attribute);

    if (colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
        const canvas = colorMapHelper.createCanvasGradient(colorMap, 1024);
        const texture = new THREE.CanvasTexture(
            canvas,
            THREE.UVMapping,
            THREE.ClampToEdgeWrapping,
            THREE.ClampToEdgeWrapping,
            THREE.NearestFilter,
            THREE.NearestFilter,
        );
        texture.needsUpdate = true;

        material.uniforms.useMap.value = 1.0;
        material.uniforms.map.value = texture;

        uvs = new Float32Array(attribute.length);

        for (let i = 0; i < attribute.length; i++) {
            uvs[i] = (attribute[i] - colorRange[0]) / (colorRange[1] - colorRange[0]);
        }

        colors = null;
    }

    const line = new MeshLine.MeshLine();

    line.setGeometry(positions, true, null, colors, uvs);
    line.geometry.computeBoundingSphere();
    line.geometry.computeBoundingBox();

    const object = new THREE.Mesh(line.geometry, material);
    object.userData.meshLine = line;
    object.userData.lastPositions = new Float32Array(positions);
    object.userData.lastUVs = uvs;
    object.userData.lastColors = verticesColors;

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    object.applyMatrix4(modelMatrix);

    object.updateMatrixWorld();

    const resizelistenerId = K3D.on(K3D.events.RESIZED, () => {
        // update outlines
        object.material.uniforms.resolution.value.x = K3D.getWorld().width;
        object.material.uniforms.resolution.value.y = K3D.getWorld().height;
    });

    object.onRemove = function () {
        K3D.off(K3D.events.RESIZED, resizelistenerId);
        if (object.material.uniforms.map.value) {
            object.material.uniforms.map.value.dispose();
            object.material.uniforms.map.value = undefined;
        }
    };

    return Promise.resolve(object);
}

function update(config, changes, obj, K3D) {
    let uvs = obj.userData.lastUVs;
    let positions = obj.userData.lastPositions;
    const colors = obj.userData.lastColors;
    const resolvedChanges = {};

    if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
        if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
            obj.material.uniforms.low.value = changes.color_range[0];
            obj.material.uniforms.high.value = changes.color_range[1];

            resolvedChanges.color_range = null;
        }

        if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries) {
            if (changes.attribute.data.length !== obj.geometry.attributes.uv.array.length) {
                return false;
            }

            uvs = new Float32Array(changes.attribute.data.length);

            for (let i = 0; i < uvs.length; i++) {
                uvs[i] = (changes.attribute.data[i] - config.color_range[0])
                    / (config.color_range[1] - config.color_range[0]);
            }

            obj.userData.lastUVs = uvs;
        }
    }

    if (typeof (changes.vertices) !== 'undefined' && !changes.vertices.timeSeries) {
        if (changes.vertices.data.length !== positions.length) {
            return false;
        }

        positions = changes.vertices.data;
        obj.userData.lastPositions = positions;
    }

    if (typeof (changes.attribute) !== 'undefined' || typeof (changes.vertices) !== 'undefined') {
        obj.userData.meshLine.setGeometry(positions, true, null, colors, uvs);
        obj.geometry.attributes.position.needsUpdate = true;

        obj.geometry.computeBoundingSphere();
        obj.geometry.computeBoundingBox();

        resolvedChanges.attribute = null;
        resolvedChanges.vertices = null;
    }

    commonUpdate(config, changes, resolvedChanges, obj, K3D);

    if (areAllChangesResolve(changes, resolvedChanges)) {
        return Promise.resolve({ json: config, obj });
    }
    return false;
}

module.exports = {
    create,
    update,
};
