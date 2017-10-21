'use strict';

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var geometry = new THREE.BufferGeometry(),
        material = new THREE.LineBasicMaterial({
            color: config.color || 0,
            linewidth: config.width || 1
        }),
        object = new THREE.Line(geometry, material),
        modelMatrix = new THREE.Matrix4(),
        position = config.vertices.buffer;

    geometry.addAttribute('position', new THREE.BufferAttribute(position, 3));

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
