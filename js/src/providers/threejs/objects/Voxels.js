const VoxelsHelper = require('../helpers/Voxels');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');

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

    update(config, changes, obj) {
        const resolvedChanges = {};

        commonUpdate(config, changes, resolvedChanges, obj);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
