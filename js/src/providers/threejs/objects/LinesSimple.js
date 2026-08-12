const THREE = require('three');
const { colorsToFloat32Array } = require('../../../core/lib/helpers/buffer');
const Fn = require('../helpers/Fn');

const { commonUpdate } = Fn;
const { areAllChangesResolve } = Fn;
const { getColorsArray } = Fn;
const { handleColorMap } = Fn;

/**
 * Loader strategy to handle Lines object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        const geometry = new THREE.BufferGeometry();
        const material = new THREE.MeshBasicMaterial({
            opacity: config.opacity,
            depthWrite: config.opacity === 1.0,
            transparent: config.opacity !== 1.0,
        });
        let verticesColors = (config.colors && config.colors.data) || null;
        const color = new THREE.Color(config.color);
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const attr = (config.attribute && config.attribute.data) || null;
        const object = new THREE.LineSegments(geometry, material);
        const modelMatrix = new THREE.Matrix4();
        const vertices = config.vertices.data;
        const indices = config.indices.data;
        const edges = new Set();

        let positions = [];
        let attribute = [];
        let colors = [];
        const edgeVertices = [];
        const jump = config.indices_type === 'segment' ? 2 : 3;
        let offsets;

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

                    edgeVertices.push(offsets[j][0], offsets[j][1]);

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
        colors = new Float32Array(colors);

        if (colorRange && colorMap && attribute.length > 0 && colorRange.length > 0
            && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, attribute, material);
        } else {
            material.setValues({ vertexColors: true });
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        object.userData.edgeVertices = new Uint32Array(edgeVertices);
        object.userData.attributeLength = attr ? attr.length : 0;
        object.userData.verticesLength = vertices.length;

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            const source = obj.userData.edgeVertices;
            const uv = obj.geometry.attributes.uv.array;

            const renormalise = (attribute, range) => {
                const low = range[0];
                const span = range[1] - low;

                for (let i = 0; i < uv.length; i++) {
                    uv[i] = (attribute[source[i]] - low) / span;
                }

                obj.geometry.attributes.uv.needsUpdate = true;
            };

            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                // A plain MeshBasicMaterial has no .uniforms; the colormap lives in the uvs.
                const attribute = (config.attribute && config.attribute.data) || null;

                if (source && attribute && attribute.length === obj.userData.attributeLength) {
                    renormalise(attribute, changes.color_range);
                    resolvedChanges.color_range = null;
                }
            }

            if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries
                && source && changes.attribute.data.length === obj.userData.attributeLength) {
                renormalise(changes.attribute.data, config.color_range);
                resolvedChanges.attribute = null;
            }
        }

        if (typeof (changes.vertices) !== 'undefined' && !changes.vertices.timeSeries
            && obj.userData.edgeVertices
            && changes.vertices.data.length === obj.userData.verticesLength) {

            const map = obj.userData.edgeVertices;
            const incoming = changes.vertices.data;
            const target = obj.geometry.attributes.position.array;

            for (let i = 0; i < map.length; i++) {
                const from = map[i] * 3;
                const to = i * 3;

                target[to] = incoming[from];
                target[to + 1] = incoming[from + 1];
                target[to + 2] = incoming[from + 2];
            }

            obj.geometry.attributes.position.needsUpdate = true;

            obj.geometry.computeBoundingSphere();
            obj.geometry.computeBoundingBox();

            resolvedChanges.vertices = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }

        return false;
    },
};
