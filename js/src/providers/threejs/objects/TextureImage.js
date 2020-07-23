'use strict';

var THREE = require('three'),
    intersectHelper = require('./../helpers/Intersection'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    commonUpdate = require('./../helpers/Fn').commonUpdate,
    buffer = require('./../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create: function (config, K3D) {
        return new Promise(function (resolve) {
            var geometry = new THREE.PlaneBufferGeometry(1, 1),
                modelMatrix = new THREE.Matrix4(),
                texture = new THREE.Texture(),
                image,
                material,
                object;

            image = document.createElement('img');
            image.src = 'data:image/' + config.file_format + ';base64,' +
                        buffer.bufferToBase64(config.binary.buffer);

            if (config.puv.data.length === 9) {
                var positionArray = geometry.attributes.position.array;

                var p = new THREE.Vector3().fromArray(config.puv.data, 0);
                var u = new THREE.Vector3().fromArray(config.puv.data, 3);
                var v = new THREE.Vector3().fromArray(config.puv.data, 6);

                p.toArray(positionArray, 0);
                p.clone().add(u).toArray(positionArray, 3);
                p.clone().add(v).toArray(positionArray, 6);
                p.clone().add(v).add(u).toArray(positionArray, 9);

                geometry.computeVertexNormals();
            }

            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            image.onload = function () {
                material = new THREE.MeshBasicMaterial({color: 0xffffff, side: THREE.DoubleSide, map: texture});
                object = new THREE.Mesh(geometry, material);

                intersectHelper.init(config, object, K3D);

                modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
                object.applyMatrix4(modelMatrix);

                object.updateMatrixWorld();

                texture.image = image;
                texture.flipY = false;
                texture.minFilter = THREE.LinearFilter;
                texture.needsUpdate = true;

                resolve(object);
            };
        });
    },

    update: function (config, changes, obj, K3D) {
        var resolvedChanges = {};

        intersectHelper.update(config, changes, resolvedChanges, obj, K3D);
        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
