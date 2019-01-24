'use strict';

var VoxelsHelper = require('./../helpers/Voxels');

function K3DVoxelsMap(config) {
    var newArray;

    this._map = new Map();

    this.set = function (x, y, z, value, updateSparseVoxels) {
        var v = config.sparse_voxels.data;

        this._map.set((z * config.space_size.data[0] + y) * config.space_size.data[1] + x, value);

        if (updateSparseVoxels) {
            for (var i = 0; i < v.length; i += 4) {
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
    create: function (config, K3D) {
        var v = config.sparse_voxels.data,
            voxels = new K3DVoxelsMap(config),
            i;

        for (i = 0; i < v.length; i += 4) {
            voxels.set(v[i], v[i + 1], v[i + 2], v[i + 3], false);
        }

        return VoxelsHelper.create(
            config,
            VoxelsHelper.generateRegularChunks(
                32, [config.space_size.data[2], config.space_size.data[1], config.space_size.data[0]], voxels
            ),
            config.space_size.data,
            K3D
        );
    }
};
