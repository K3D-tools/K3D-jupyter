'use strict';

/**
 * Loader strategy to handle STL object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var loader = new THREE.STLLoader(),
        modelMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.color,
            emissive: 0x072534,
            shading: THREE.FlatShading,
            side: THREE.DoubleSide
        }),
        text = config.text,
        binary = config.binary,
        geometry,
        object;

    if (binary && binary.byteLength > 0) {
        geometry = loader.parseBinary(binary.buffer);
    } else {
        geometry = loader.parseASCII(text);
    }

    if (geometry.hasColors) {
        material = new THREE.MeshPhongMaterial({opacity: geometry.alpha, vertexColors: THREE.VertexColors});
    }

    object = new THREE.Mesh(geometry, material);

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelMatrix.set.apply(modelMatrix, config.model_matrix);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
