'use strict';

var VoxelsHelper = require('./../helpers/Voxels');

function K3DVoxelsMap(group) {
    var cx = group.coord.data[0],
        cy = group.coord.data[1],
        cz = group.coord.data[2],
        sx = group.voxels.shape[2],
        sy = group.voxels.shape[1],
        sz = group.voxels.shape[0],
        data = group.voxels.data;

    function dist(a) {
        var dx = group.coord.data[0] - a.offset[0],
            dy = group.coord.data[1] - a.offset[1],
            dz = group.coord.data[2] - a.offset[2];

        dx = (dx < 0 ? 1000 : dx);
        dy = (dy < 0 ? 1000 : dy);
        dz = (dz < 0 ? 1000 : dz);

        return dx + dy + dz;
    }

    this.neighbours = [];

    this.set = function (x, y, z, value, skipSearchingNeighbours) {
        var lx = x - cx,
            ly = y - cy,
            lz = z - cz;

        if (lx >= 0 && lx < sx &&
            ly >= 0 && ly < sy &&
            lz >= 0 && lz < sz) {
            data[(lz * sy + ly) * sx + lx] = value;
        } else if (!skipSearchingNeighbours) {
            for (var i = 0; i < this.neighbours.length; i++) {
                if (this.neighbours[i].voxels.set(x, y, z, value, true)) {
                    return true;
                }
            }
        }

        return true;
    };

    this.get = function (x, y, z, skipSearchingNeighbours) {
        var lx = x - cx,
            ly = y - cy,
            lz = z - cz;

        if (lx >= 0 && lx < sx &&
            ly >= 0 && ly < sy &&
            lz >= 0 && lz < sz) {
            return data[(lz * sy + ly) * sx + lx];
        } else if (!skipSearchingNeighbours) {
            for (var i = 0; i < this.neighbours.length; i++) {
                var v = this.neighbours[i].voxels.get(x, y, z, true);

                if (v !== -1) {
                    return v;
                }
            }
        }

        return -1;
    };

    this.setNeighbours = function (list, chunk) {
        this.neighbours = list.reduce(function (p, v) {
            if (v === chunk) {
                return p;
            }

            var wx = Math.max(group.coord.data[0] + group.voxels.shape[2], v.offset[0] + v.size[0]) -
                     Math.min(group.coord.data[0], v.offset[0]),

                wy = Math.max(group.coord.data[1] + group.voxels.shape[1], v.offset[1] + v.size[1]) -
                     Math.min(group.coord.data[1], v.offset[1]),

                wz = Math.max(group.coord.data[2] + group.voxels.shape[0], v.offset[2] + v.size[2]) -
                     Math.min(group.coord.data[2], v.offset[2]),

                rx = group.voxels.shape[2] + v.size[0],
                ry = group.voxels.shape[1] + v.size[1],
                rz = group.voxels.shape[0] + v.size[2];

            if (wx <= rx && wy <= ry && wz <= rz) {
                if ((wx === rx) && (wy === ry) && (wz === rz)) {
                    return p;
                }

                p.push(v);
            }

            return p;
        }, []);

        this.neighbours.sort(function (a, b) {
            return dist(a) - dist(b);
        });
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
        var chunkList = [];

        config.voxels_group.forEach(function (group) {
            chunkList.push({
                voxels: new K3DVoxelsMap(group),
                size: [group.voxels.shape[2], group.voxels.shape[1], group.voxels.shape[0]],
                offset: group.coord.data,
                multiple: group.multiple
            });
        });

        chunkList.forEach(function (chunk) {
            chunk.voxels.setNeighbours(chunkList, chunk);
        });

        return VoxelsHelper.create(
            config,
            chunkList,
            config.space_size.data,
            K3D
        );
    }
};
