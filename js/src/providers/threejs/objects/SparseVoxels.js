const VoxelsHelper = require('../helpers/Voxels');
const { areAllChangesResolve } = require('../helpers/Fn');
const { commonUpdate } = require('../helpers/Fn');

function K3DVoxelsMap(config) {
    let newArray;

    this._map = new Map();

    this.set = function (x, y, z, value, updateSparseVoxels) {
        const v = config.sparse_voxels.data;

        this._map.set((z * config.space_size.data[0] + y) * config.space_size.data[1] + x, value);

        if (updateSparseVoxels) {
            for (let i = 0; i < v.length; i += 4) {
                if (v[i] === x && v[i + 1] === y && v[i + 2] === z) {
                    if (value === 0) {
                        // Remove
                        newArray = new v.constructor(v.length - 4);

                        newArray.set(v.subarray(0, i));
                        newArray.set(v.subarray(i + 4, v.length), i);
                        config.sparse_voxels.data = newArray;
                        config.sparse_voxels.shape[0]--;
                    } else {
                        // Change
                        v[i + 3] = value;
                    }

                    return;
                }
            }

            // Add new
            if (value !== 0) {
                newArray = new v.constructor(v.length + 4);

                newArray.set(v);
                newArray.set([x, y, z, value], v.length);
                config.sparse_voxels.data = newArray;
                config.sparse_voxels.shape[0]++;
            }
        }
    };

    this.get = function (x, y, z) {
        return this._map.get((z * config.space_size.data[0] + y) * config.space_size.data[1] + x);
    };
}

/**
 * Loader strategy to handle SparseVoxels object
 * @method Voxel
 * @memberof K3D.Providers.ThreeJS.Objects
 * @param {Object} config all configurations params from JSON
 * @param {Object} K3D
 */
module.exports = {
    create(config, K3D) {
        const v = config.sparse_voxels.data;
        const voxels = new K3DVoxelsMap(config);
        let i;

        for (i = 0; i < v.length; i += 4) {
            voxels.set(v[i], v[i + 1], v[i + 2], v[i + 3], false);
        }

        return VoxelsHelper.create(
            config,
            VoxelsHelper.generateRegularChunks(
                32, [config.space_size.data[2], config.space_size.data[1], config.space_size.data[0]], voxels,
            ),
            config.space_size.data,
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
            return Promise.resolve({ json: config, obj });
        }
        return false;
    },
};
