'use strict';

var buffer = require('./../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle STL object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var loader = new THREE.STLLoader(),
        modelViewMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.get('color'),
            emissive: 0x072534,
            shading: THREE.FlatShading
        }),
        STL = config.get('STL'),
        geometry,
        object;

    try {
        //Check if string is in base64 format
        STL = buffer.base64ToArrayBuffer(STL);
        geometry = loader.parseBinary(STL);
    } catch (e) {
        //plain string with STL
        geometry = loader.parseASCII(STL);
    }

    if (geometry.hasColors) {
        material = new THREE.MeshPhongMaterial({opacity: geometry.alpha, vertexColors: THREE.VertexColors});
    }

    object = new THREE.Mesh(geometry, material);

    geometry.computeBoundingSphere();

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
