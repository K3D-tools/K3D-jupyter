'use strict';

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {K3D.Config} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = function (config) {
    return new Promise(function (resolve) {
        var geometry = new THREE.PlaneBufferGeometry(1, 1),
            modelMatrix = new THREE.Matrix4(),
            texture = new THREE.Texture(),
            image,
            material,
            object;

        image = document.createElement('img');
        image.src = config.get('image');

        geometry.computeBoundingSphere();
        geometry.computeBoundingBox();

        image.onload = function () {
            material = new THREE.MeshBasicMaterial({color: 0xffffff, side: THREE.DoubleSide, map: texture});
            object = new THREE.Mesh(geometry, material);

            modelMatrix.set.apply(modelMatrix, config.get('modelMatrix'));
            object.applyMatrix(modelMatrix);

            object.updateMatrixWorld();

            texture.image = image;
            texture.minFilter = THREE.LinearFilter;
            texture.needsUpdate = true;

            resolve(object);
        };
    });
};
