const THREE = require('three');
const Fn = require('../helpers/Fn');

const {areAllChangesResolve} = Fn;
const {commonUpdate} = Fn;
const {colorsToFloat32Array} = require('../../../core/lib/helpers/buffer');
const streamLine = require('../helpers/Streamline');

const {handleColorMap} = Fn;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config) {
        config.radial_segments = typeof (config.radial_segments) !== 'undefined' ? config.radial_segments : 8;
        config.width = typeof (config.width) !== 'undefined' ? config.width : 0.1;
        config.opacity = typeof (config.opacity) !== 'undefined' ? config.opacity : 1.0;

        const material = new THREE.MeshPhongMaterial({
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            wireframe: false,
            opacity: config.opacity,
            depthWrite: config.opacity === 1.0,
            transparent: config.opacity !== 1.0,
        });
        const radialSegments = config.radial_segments;
        const {width} = config;
        let verticesColors = (config.colors && config.colors.data) || null;
        const color = new THREE.Color(config.color);
        const colorRange = config.color_range;
        const colorMap = (config.color_map && config.color_map.data) || null;
        const attribute = (config.attribute && config.attribute.data) || null;
        const modelMatrix = new THREE.Matrix4();
        const position = config.vertices.data;

        if (verticesColors && verticesColors.length === position.length / 3) {
            verticesColors = colorsToFloat32Array(verticesColors);
        }

        const geometry = streamLine(position, attribute, width, radialSegments, color, verticesColors, colorRange);

        if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0
            && colorMap.length > 0) {
            handleColorMap(geometry, colorMap, colorRange, null, material);
        } else {
            material.setValues({vertexColors: THREE.VertexColors});
        }

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        modelMatrix.set.apply(modelMatrix, config.model_matrix.data);

        const object = new THREE.Mesh(geometry, material);
        object.applyMatrix4(modelMatrix);

        object.updateMatrixWorld();

        return Promise.resolve(object);
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
