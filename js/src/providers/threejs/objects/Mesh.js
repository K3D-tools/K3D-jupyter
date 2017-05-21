'use strict';

var buffer = require('./../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Mesh object
 * @method STL
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var modelViewMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.get('color'),
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        }),
        vertices = config.get('vertices'),
        indices = config.get('indices'),
        geometry = new THREE.BufferGeometry(),
        object;

    if (typeof (vertices) === 'string') {
        vertices = buffer.base64ToArrayBuffer(vertices);
    }

    if (typeof (indices) === 'string') {
        indices = buffer.base64ToArrayBuffer(indices);
    }

    geometry.addAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    object = new THREE.Mesh(geometry, material);

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));
    object.applyMatrix(modelViewMatrix);

    object.updateMatrixWorld();

    return Promise.resolve(object);
};
