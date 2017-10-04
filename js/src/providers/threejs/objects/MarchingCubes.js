'use strict';

var marchingCubesPolygonise = require('./../../../core/lib/helpers/marchingCubesPolygonise');
/**
 * Loader strategy to handle Marching Cubes object
 * @method MarchingCubes
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {

    var scalarField = config.scalar_field.buffer,
        width = config.scalar_field.shape[2],
        height = config.scalar_field.shape[1],
        length = config.scalar_field.shape[0],
        level = config.level,
        modelMatrix = new THREE.Matrix4(),
        material = new THREE.MeshPhongMaterial({
            color: config.color,
            emissive: 0,
            shininess: 25,
            specular: 0x111111,
            side: THREE.DoubleSide,
            shading: THREE.FlatShading
        }),
        geometry = new THREE.BufferGeometry(),
        positions = [],
        object,
        x, y, z,
        polygonise = marchingCubesPolygonise;

    for (z = 0; z < length - 1; z++) {
        for (y = 0; y < height - 1; y++) {
            for (x = 0; x < width - 1; x++) {
                polygonise(positions, scalarField, width, height, length, level, x, y, z);
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
        new THREE.Vector3(0.0, 0.0, 0.0),
        new THREE.Vector3(1.0, 1.0, 1.0)
    );

    object = new THREE.Mesh(geometry, material);

    modelMatrix.set.apply(modelMatrix, config.model_matrix.buffer);

    object.position.set(-0.5, -0.5, -0.5);
    object.updateMatrix();

    object.applyMatrix(modelMatrix);
    object.updateMatrixWorld();

    return Promise.resolve(object);
};
