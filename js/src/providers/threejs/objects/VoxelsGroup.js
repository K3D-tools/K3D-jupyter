'use strict';
var VoxelsHelper = require('./../helpers/Voxels'),
    areAllChangesResolve = require('./../helpers/Fn').areAllChangesResolve,
    modelMatrixUpdate = require('./../helpers/Fn').modelMatrixUpdate;

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
    this.data = data;

    this.set = function (x, y, z, value, skipSearchingNeighbours) {
        var lx = x - cx,
            ly = y - cy,
            lz = z - cz;

        if (lx >= 0 && lx < sx &&
            ly >= 0 && ly < sy &&
            lz >= 0 && lz < sz) {
            this.data[(lz * sy + ly) * sx + lx] = value;
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
            return this.data[(lz * sy + ly) * sx + lx];
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
            if (v.id === chunk.id) {
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

function prepareChunk(group, idx) {
    return {
        voxels: new K3DVoxelsMap(group),
        size: [group.voxels.shape[2], group.voxels.shape[1], group.voxels.shape[0]],
        offset: group.coord.data,
        multiple: group.multiple,
        id: group.id || idx,
        idx: idx
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

        if (typeof (config.voxels_group) !== 'undefined') {
            config.voxels_group.forEach(function (group, idx) {
                chunkList.push(prepareChunk(group, idx));
            });
        }

        if (typeof (config.chunks_ids) !== 'undefined') {
            config.chunks_ids.forEach(function (chunkId, idx) {
                chunkList.push(prepareChunk(K3D.getWorld().chunkList[chunkId].attributes, idx));
            });
        }

        chunkList.forEach(function (chunk) {
            chunk.voxels.setNeighbours(chunkList, chunk);
        });

        return VoxelsHelper.create(
            config,
            chunkList,
            config.space_size.data,
            K3D
        );
    },

    update: function (config, changes, obj, K3D) {
        if (typeof(changes.opacity) !== 'undefined' && !changes.opacity.timeSeries) {
            obj.traverse(function (object) {
                if (object.material) {
                    if (object.material.userData.outline) {
                        object.material.opacity = object.material.uniforms.opacity.value = config.opacity * 0.75;
                    } else {
                        object.material.opacity = config.opacity;
                        object.material.depthWrite = config.opacity === 1.0;
                        object.material.transparent = config.opacity !== 1.0;
                    }
                }
            });

            changes.opacity = null;
        }

        if (typeof(changes.chunks_ids) !== 'undefined' && !changes.chunks_ids.timeSeries) {
            var idsMap = {}, affectedIds = new Set();

            obj.children.forEach(function (g) {
                if (g.voxel) {
                    idsMap[g.voxel.chunk.id] = g;
                }
            });

            var ids = Object.keys(idsMap).reduce(function (p, v) {
                p.push(parseInt(v));
                return p;
            }, []);

            // Remove
            _.difference(ids, changes.chunks_ids).forEach(function (id) {
                idsMap[id].voxel.chunk.voxels.neighbours.forEach(function (value) {
                    affectedIds.add(value.id);
                });

                obj.remove(idsMap[id]);
                obj.voxelsChunks = obj.voxelsChunks.filter(function (v) {
                    return v.id !== id;
                });

                idsMap[id].children.forEach(function (mesh) {
                    mesh.geometry.dispose();
                    mesh.material.dispose();
                });
                idsMap[id] = null;
            });

            // Add new
            _.difference(changes.chunks_ids, ids).forEach(function (id) {
                var chunk = prepareChunk(K3D.getWorld().chunkList[id].attributes, obj.voxelsChunks.length),
                    mesh = obj.addChunk(chunk);

                obj.voxelsChunks.push(chunk);
                idsMap[id] = mesh;
            });

            obj.voxelsChunks.forEach(function (chunk) {
                chunk.voxels.setNeighbours(obj.voxelsChunks, chunk);
            });

            _.difference(changes.chunks_ids, ids).forEach(function (id) {
                var chunk = idsMap[id].voxel.chunk;

                affectedIds.add(chunk.id);
                chunk.voxels.neighbours.forEach(function (value) {
                    affectedIds.add(value.id);
                });
            });

            // Update
            for (var id of affectedIds.values()) {
                if (idsMap[id]) {
                    obj.updateChunk(idsMap[id].voxel.chunk, true);
                }
            }

            changes.chunks_ids = null;
        }

        if (typeof(changes._hold_remeshing) !== 'undefined' && !changes._hold_remeshing.timeSeries) {
            obj.holdRemeshing = config._hold_remeshing;

            if (!config._hold_remeshing) {
                obj.rebuildChunk();
            }

            changes._hold_remeshing = null;
        }

        modelMatrixUpdate(config, changes, obj);

        if (areAllChangesResolve(changes)) {
            return Promise.resolve({json: config, obj: obj});
        } else {
            return false;
        }
    }
};
