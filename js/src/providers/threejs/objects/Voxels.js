const VoxelsHelper = require('../helpers/Voxels');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');

/**
 * Loader strategy to handle SparseVoxels object
 * @method Voxel
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {Object} K3D
 */
module.exports = {
    create(config, K3D) {
        return VoxelsHelper.create(
            config,
            VoxelsHelper.generateRegularChunks(96, config.voxels.shape, config.voxels.data),
            [config.voxels.shape[2], config.voxels.shape[1], config.voxels.shape[0]],
            K3D,
        );
    },

    update(config, changes, obj, K3D) {
        const resolvedChanges = {};

        if (typeof (changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.traverse((object) => {
                if (object.material) {
                    if (object.material.userData.outline) {
                        object.material.uniforms.opacity.value = config.opacity * 0.75;
                        object.material.opacity = object.material.uniforms.opacity.value;
                    } else {
                        object.material.opacity = config.opacity;
                        object.material.depthWrite = config.opacity === 1.0;
                        object.material.transparent = config.opacity !== 1.0;
                    }
                }
            });

            resolvedChanges.opacity = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }
        return false;
    },
};
