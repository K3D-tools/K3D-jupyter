const THREE = require('three');
const intersectHelper = require('../helpers/Intersection');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');
const buffer = require('../../../core/lib/helpers/buffer');

/**
 * Loader strategy to handle Texture object
 * @method Line
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @return {Object} 3D object ready to render
 */
module.exports = {
    create(config, K3D) {
        return new Promise((resolve) => {
            const geometry = new THREE.PlaneBufferGeometry(1, 1);
            const modelMatrix = new THREE.Matrix4();
            const texture = new THREE.Texture();
            let material;
            let object;

            config.interpolation = typeof (config.interpolation) !== 'undefined' ? config.interpolation : true;

            const image = document.createElement('img');
            image.src = `data:image/${config.file_format};base64,${
                buffer.bufferToBase64(config.binary.data.buffer)}`;

            if (config.puv.data.length === 9) {
                const positionArray = geometry.attributes.position.array;

                const p = new THREE.Vector3().fromArray(config.puv.data, 0);
                const u = new THREE.Vector3().fromArray(config.puv.data, 3);
                const v = new THREE.Vector3().fromArray(config.puv.data, 6);

                p.toArray(positionArray, 0);
                p.clone().add(u).toArray(positionArray, 3);
                p.clone().add(v).toArray(positionArray, 6);
                p.clone().add(v).add(u).toArray(positionArray, 9);

                geometry.computeVertexNormals();
            }

            geometry.computeBoundingSphere();
            geometry.computeBoundingBox();

            image.onload = function () {
                material = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, map: texture });
                object = new THREE.Mesh(geometry, material);

                intersectHelper.init(config, object, K3D);

                modelMatrix.set.apply(modelMatrix, config.model_matrix.data);
                object.applyMatrix4(modelMatrix);

                object.updateMatrixWorld();

                if (config.interpolation) {
                    texture.minFilter = THREE.LinearFilter;
                    texture.magFilter = THREE.LinearFilter;
                } else {
                    texture.minFilter = THREE.NearestFilter;
                    texture.magFilter = THREE.NearestFilter;
                }

                texture.image = image;
                texture.flipY = false;
                texture.minFilter = THREE.LinearFilter;
                texture.needsUpdate = true;

                resolve(object);
            };
        });
    },

    update(config, changes, obj) {
        const resolvedChanges = {};

        intersectHelper.update(config, changes, resolvedChanges, obj);
        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
