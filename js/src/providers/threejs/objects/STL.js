'use strict';

/**
 * Loader strategy to handle STL object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    config.visible = typeof(config.visible) !== 'undefined' ? config.visible : true;
    config.color = typeof(config.color) !== 'undefined' ? config.color : 255;
    config.wireframe = typeof(config.wireframe) !== 'undefined' ? config.wireframe : false;

    var loader = new THREE.STLLoader(),
        modelMatrix = new THREE.Matrix4(),
        MaterialConstructor = config.wireframe ? THREE.MeshBasicMaterial : THREE.MeshPhongMaterial,
        material = new MaterialConstructor({
            color: config.color,
            emissive: 0x072534,
            flatShading: true,
            side: THREE.DoubleSide,
            wireframe: config.wireframe
        }),
        text = config.text,
        binary = config.binary,
        geometry,
        object;

    if (text === null || typeof(text) === 'undefined') {
        if (typeof(binary.buffer.buffer) !== 'undefined') {
            geometry = loader.parse(binary.buffer.buffer);
        } else {
            geometry = loader.parse(binary.buffer);
        }
    } else {
        geometry = loader.parse(text);
    }

    if (geometry.hasColors) {
        material = new THREE.MeshPhongMaterial({
            opacity: geometry.alpha,
            vertexColors: THREE.VertexColors,
            wireframe: config.wireframe
        });
    }

    object = new THREE.Mesh(geometry, material);

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);
    object.applyMatrix(modelMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
