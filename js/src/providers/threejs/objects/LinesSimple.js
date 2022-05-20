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
        const material = new THREE.MeshBasicMaterial();
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
        let jump = config.indices_type === 'segment' ? 2 : 3;
        let offsets;

        let verticesCount = vertices.length / 3;

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
                    [indices[i], indices[i + 1]]
                ];
            }

            for (let j = 0; j < offsets.length; j++) {
                let hash = offsets[j][0] > offsets[j][1]
                    ? offsets[j][0] + offsets[j][1] * verticesCount
                    : offsets[j][1] + offsets[j][0] * verticesCount;

                if (!edges.has(hash)) {
                    edges.add(hash);

                    let o1 = offsets[j][0] * 3;
                    let o2 = offsets[j][1] * 3;

                    positions.push(
                        vertices[o1], vertices[o1 + 1], vertices[o1 + 2],
                        vertices[o2], vertices[o2 + 1], vertices[o2 + 2]
                    );

                    if (verticesColors && verticesColors.length > 0) {
                        colors.push(
                            verticesColors[o1], verticesColors[o1 + 1], verticesColors[o1 + 2],
                            verticesColors[o2], verticesColors[o2 + 1], verticesColors[o2 + 2]
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
            material.setValues({ vertexColors: THREE.VertexColors });
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        if (typeof (obj.geometry.attributes.uv) !== 'undefined') {
            if (typeof (changes.color_range) !== 'undefined' && !changes.color_range.timeSeries) {
                obj.material.uniforms.low.value = changes.color_range[0];
                obj.material.uniforms.high.value = changes.color_range[1];

                resolvedChanges.color_range = null;
            }

            if (typeof (changes.attribute) !== 'undefined' && !changes.attribute.timeSeries
                && changes.attribute.data.length === obj.geometry.attributes.uv.array.length) {
                const data = obj.geometry.attributes.uv.array;

                for (let i = 0; i < data.length; i++) {
                    data[i] = (changes.attribute.data[i] - config.color_range[0])
                        / (config.color_range[1] - config.color_range[0]);
                }

                obj.geometry.attributes.uv.needsUpdate = true;
                resolvedChanges.attribute = null;
            }
        }

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }

        return false;
    },
};
