'use strict';

var buffer = require('./../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Line object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
*/
module.exports = function (config) {

    var geometry = new THREE.BufferGeometry(),
        material = new THREE.LineBasicMaterial({
            color: config.get('color', 0),
            linewidth: config.get('lineWidth', 1)
        }),
        object = new THREE.Line(geometry, material),
        modelViewMatrix = new THREE.Matrix4(),
        position = config.get('pointsPositions');

    if (typeof (position) === 'string') {
        position = buffer.base64ToArrayBuffer(position);
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(buffer.toFloat32Array(position), 3));

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
