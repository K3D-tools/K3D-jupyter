'use strict';

var Fn = require('./../helpers/Fn'),
    colorsToFloat32Array = require('./../../../core/lib/helpers/buffer').colorsToFloat32Array,
    streamLine = require('./../helpers/Streamline'),
    handleColorMap = Fn.handleColorMap;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var geometry,
        material = new THREE.MeshPhongMaterial({
            emissive: 0,
            shininess: 25,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.SmoothShading,
            wireframe: false
        }),
        radialSegments = config.radial_segments || 8,
        width = config.width || 0.1,
        verticesColors = (config.colors && config.colors.buffer) || null,
        color = typeof(config.color) !== 'number' || config.color < 0 || config.color > 0xffffff ?
            new THREE.Color(0xff00) : new THREE.Color(config.color),
        colorRange = config.color_range,
        colorMap = (config.color_map && config.color_map.buffer) || null,
        attribute = (config.attribute && config.attribute.buffer) || null,
        object,
        modelMatrix = new THREE.Matrix4(),
        position = config.vertices.buffer;

    if (verticesColors && verticesColors.length === position.length / 3) {
        verticesColors = colorsToFloat32Array(verticesColors);
    }

    geometry = streamLine(position, attribute, width, radialSegments, color, verticesColors, colorRange);

    if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
        handleColorMap(geometry, colorMap, colorRange, null, material);
    } else {
        material.setValues({vertexColors: THREE.VertexColors});
    }

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);

    object = new THREE.Mesh(geometry, material);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
