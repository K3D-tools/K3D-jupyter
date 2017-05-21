'use strict';

var buffer = require('./../../../core/lib/helpers/buffer'),
    marchingCubesPolygonise = require('./../../../core/lib/helpers/marchingCubesPolygonise');
/**
 * Loader strategy to handle Marching Cubes object
 * @method MarchingCubes
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var scalarsField = config.get('scalarsField'),
        width = config.get('width'),
        height = config.get('height'),
        length = config.get('length'),
        level = config.get('level'),
        modelViewMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.get('color'),
            emissive: 0,
            shininess: 50,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        }),
        geometry = new THREE.BufferGeometry(),
        positions = [],
        object,
        x, y, z,
        polygonise = marchingCubesPolygonise,
        toFloat32Array = buffer.toFloat32Array;

    if (typeof (scalarsField) === 'string') {
        scalarsField = buffer.base64ToArrayBuffer(scalarsField);
    }

    scalarsField = toFloat32Array(scalarsField);

    for (z = 0; z < length - 1; z++) {
        for (y = 0; y < height - 1; y++) {
            for (x = 0; x < width - 1; x++) {
                polygonise(positions, scalarsField, width, height, length, level, x, y, z);
            }
        }
    }

    positions = new Float32Array(positions);
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));

    geometry.boundingSphere = new THREE.Sphere(
        new THREE.Vector3(0.5, 0.5, 0.5),
        new THREE.Vector3(0.5, 0.5, 0.5).length()
    );

    geometry.boundingBox = new THREE.Box3(
        new THREE.Vector3(-0.5, -0.5, -0.5),
        new THREE.Vector3(0.5, 0.5, 0.5)
    );

    object = new THREE.Mesh(geometry, material);

    modelViewMatrix.set.apply(modelViewMatrix, config.get('modelViewMatrix'));

    object.position.set(-0.5, -0.5, -0.5);
    object.updateMatrix();

    object.applyMatrix(modelViewMatrix);
    object.updateMatrixWorld();

    return Promise.resolve(object);
};
