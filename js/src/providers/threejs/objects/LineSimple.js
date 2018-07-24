'use strict';

var colorsToFloat32Array = require('./../../../core/lib/helpers/buffer').colorsToFloat32Array,
    Fn = require('./../helpers/Fn'),
    getColorsArray = Fn.getColorsArray,
    handleColorMap = Fn.handleColorMap;

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var geometry = new THREE.BufferGeometry(),
        material = new THREE.MeshBasicMaterial(),
        verticesColors = (config.colors && config.colors.data) || null,
        color = new THREE.Color(config.color),
        colors,
        colorRange = config.color_range,
        colorMap = (config.color_map && config.color_map.data) || null,
        attribute = (config.attribute && config.attribute.data) || null,
        object = new THREE.Line(geometry, material),
        modelMatrix = new THREE.Matrix4(),
        position = config.vertices.data;

    if (attribute && colorRange && colorMap && attribute.length > 0 && colorRange.length > 0 && colorMap.length > 0) {
        handleColorMap(geometry, colorMap, colorRange, attribute, material);
    } else {
        colors = ( verticesColors && verticesColors.length === position.length / 3 ?
                colorsToFloat32Array(verticesColors) : getColorsArray(color, position.length / 3)
        );

        material.setValues({vertexColors: THREE.VertexColors});
        geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(position, 3));

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
