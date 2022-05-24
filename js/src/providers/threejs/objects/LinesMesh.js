const THREE = require('three');
const Fn = require('../helpers/Fn');

const { areAllChangesResolve } = Fn;
const { commonUpdate } = Fn;
const { colorsToFloat32Array } = require('../../../core/lib/helpers/buffer');
const streamLine = require('../helpers/Streamline');
const BufferGeometryUtils = require('three/examples/jsm/utils/BufferGeometryUtils');

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
        config.radial_segments = typeof (config.radial_segments) !== 'undefined' ? config.radial_segments : 8;
        config.width = typeof (config.width) !== 'undefined' ? config.width : 0.1;

        const material = new THREE.MeshPhongMaterial({
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            wireframe: false,
        });
        const radialSegments = config.radial_segments;
        const { width } = config;
        let verticesColors = (config.colors && config.colors.data) || null;
        const color = new THREE.Color(config.color);
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const attribute = (config.attribute && config.attribute.data) || [];
        const modelMatrix = new THREE.Matrix4();
        const position = config.vertices.data;
        const indices = config.indices.data;
        const edges = new Set();
        let jump = config.indices_type == 'segment' ? 2 : 3;

        if (verticesColors && verticesColors.length === position.length / 3) {
            verticesColors = colorsToFloat32Array(verticesColors);
        }

        let g = [];
        let verticesCount = position.length / 3;
        let offsets;

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

                    g.push(
                        streamLine(
                            [
                                position[o1], position[o1 + 1], position[o1 + 2],
                                position[o2], position[o2 + 1], position[o2 + 2]
                            ],
                            attribute.length > 0 ? [attribute[offsets[j][0]], attribute[offsets[j][1]]] : null,
                            width,
                            radialSegments,
                            color,
                            verticesColors.length > 0
                                ? [verticesColors[o1], verticesColors[o1 + 1], verticesColors[o1 + 2],
                                    verticesColors[o2], verticesColors[o2 + 1], verticesColors[o2 + 2]]
                                : null,
                            colorRange
                        )
                    );
                }
            }
        }

        let geometry = BufferGeometryUtils.mergeBufferGeometries(g);

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0
            && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, null, material);
        } else {
            material.setValues({ vertexColors: THREE.VertexColors });
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

        const object = new THREE.Mesh(geometry, material);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
