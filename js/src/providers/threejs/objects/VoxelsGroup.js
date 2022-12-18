const VoxelsHelper = require('../helpers/Voxels');
const _ = require('../../../lodash');
const {areAllChangesResolve} = require('../helpers/Fn');
const {commonUpdate} = require('../helpers/Fn');

function K3DVoxelsMap(group) {
    const cx = group.coord.data[0];
    const cy = group.coord.data[1];
    const cz = group.coord.data[2];
    const sx = group.voxels.shape[2];
    const sy = group.voxels.shape[1];
    const sz = group.voxels.shape[0];
    const {data} = group.voxels;

    function dist(a) {
        let dx = group.coord.data[0] - a.offset[0];
        let dy = group.coord.data[1] - a.offset[1];
        let dz = group.coord.data[2] - a.offset[2];

        dx = (dx < 0 ? 1000 : dx);
        dy = (dy < 0 ? 1000 : dy);
        dz = (dz < 0 ? 1000 : dz);

        return dx + dy + dz;
    }

    this.neighbours = [];
    this.data = data;

    this.set = function (x, y, z, value, skipSearchingNeighbours) {
        const lx = x - cx;
        const ly = y - cy;
        const lz = z - cz;

        if (lx >= 0 && lx < sx
            && ly >= 0 && ly < sy
            && lz >= 0 && lz < sz) {
            this.data[(lz * sy + ly) * sx + lx] = value;
        } else if (!skipSearchingNeighbours) {
            for (let i = 0; i < this.neighbours.length; i++) {
                if (this.neighbours[i].voxels.set(x, y, z, value, true)) {
                    return true;
                }
            }
        }

        return true;
    };

    this.get = function (x, y, z, skipSearchingNeighbours) {
        const lx = x - cx;
        const ly = y - cy;
        const lz = z - cz;

        if (lx >= 0 && lx < sx
            && ly >= 0 && ly < sy
            && lz >= 0 && lz < sz) {
            return this.data[(lz * sy + ly) * sx + lx];
        }
        if (!skipSearchingNeighbours) {
            for (let i = 0; i < this.neighbours.length; i++) {
                const v = this.neighbours[i].voxels.get(x, y, z, true);

                if (v !== -1) {
                    return v;
                }
            }
        }

        return -1;
    };

    this.setNeighbours = function (list, chunk) {
        this.neighbours = list.reduce((p, v) => {
            if (v.id === chunk.id) {
                return p;
            }

            const wx = Math.max(group.coord.data[0] + group.voxels.shape[2], v.offset[0] + v.size[0])
                - Math.min(group.coord.data[0], v.offset[0]);

            const wy = Math.max(group.coord.data[1] + group.voxels.shape[1], v.offset[1] + v.size[1])
                - Math.min(group.coord.data[1], v.offset[1]);

            const wz = Math.max(group.coord.data[2] + group.voxels.shape[0], v.offset[2] + v.size[2])
                - Math.min(group.coord.data[2], v.offset[2]);

            const rx = group.voxels.shape[2] + v.size[0];
            const ry = group.voxels.shape[1] + v.size[1];
            const rz = group.voxels.shape[0] + v.size[2];

            if (wx <= rx && wy <= ry && wz <= rz) {
                if ((wx === rx) && (wy === ry) && (wz === rz)) {
                    return p;
                }

                p.push(v);
            }

            return p;
        }, []);

        this.neighbours.sort((a, b) => dist(a) - dist(b));
    };
}

function prepareChunk(group, idx) {
    return {
        voxels: new K3DVoxelsMap(group),
        size: [group.voxels.shape[2], group.voxels.shape[1], group.voxels.shape[0]],
        offset: group.coord.data,
        multiple: group.multiple,
        id: group.id || idx,
        idx,
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
        const chunkList = [];

        if (typeof (config.voxels_group) !== 'undefined') {
            config.voxels_group.forEach((group, idx) => {
                chunkList.push(prepareChunk(group, idx));
            });
        }

        if (typeof (config.chunks_ids) !== 'undefined') {
            config.chunks_ids.forEach((chunkId, idx) => {
                chunkList.push(prepareChunk(K3D.getWorld().chunkList[chunkId].attributes, idx));
            });
        }

        chunkList.forEach((chunk) => {
            chunk.voxels.setNeighbours(chunkList, chunk);
        });

        return VoxelsHelper.create(
            config,
            chunkList,
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

        if (typeof (changes.chunks_ids) !== 'undefined' && !changes.chunks_ids.timeSeries) {
            const idsMap = {};
            const affectedIds = new Set();

            obj.children.forEach((g) => {
                if (g.voxel) {
                    idsMap[g.voxel.chunk.id] = g;
                }
            });

            const ids = Object.keys(idsMap).reduce((p, v) => {
                p.push(parseInt(v, 10));
                return p;
            }, []);

            // Remove
            _.difference(ids, changes.chunks_ids).forEach((id) => {
                idsMap[id].voxel.chunk.voxels.neighbours.forEach((value) => {
                    affectedIds.add(value.id);
                });

                obj.remove(idsMap[id]);
                obj.voxelsChunks = obj.voxelsChunks.filter((v) => v.id !== id);

                idsMap[id].children.forEach((mesh) => {
                    mesh.geometry.dispose();
                    mesh.material.dispose();
                });
                idsMap[id] = null;
            });

            // Add new
            _.difference(changes.chunks_ids, ids).forEach((id) => {
                const chunk = prepareChunk(K3D.getWorld().chunkList[id].attributes, obj.voxelsChunks.length);
                const mesh = obj.addChunk(chunk);

                obj.voxelsChunks.push(chunk);
                idsMap[id] = mesh;
            });

            obj.voxelsChunks.forEach((chunk) => {
                chunk.voxels.setNeighbours(obj.voxelsChunks, chunk);
            });

            _.difference(changes.chunks_ids, ids).forEach((id) => {
                const {chunk} = idsMap[id].voxel;

                affectedIds.add(chunk.id);
                chunk.voxels.neighbours.forEach((value) => {
                    affectedIds.add(value.id);
                });
            });

            // Update
            affectedIds.forEach((id) => {
                if (idsMap[id]) {
                    obj.updateChunk(idsMap[id].voxel.chunk, true);
                }
            });

            resolvedChanges.chunks_ids = null;
        }

        if (typeof (changes._hold_remeshing) !== 'undefined' && !changes._hold_remeshing.timeSeries) {
            obj.holdRemeshing = config._hold_remeshing;

            if (!config._hold_remeshing) {
                obj.rebuildChunk();
            }

            resolvedChanges._hold_remeshing = null;
        }

        commonUpdate(config, changes, resolvedChanges, obj, K3D);

        if (areAllChangesResolve(changes, resolvedChanges)) {
            return Promise.resolve({json: config, obj});
        }

        return false;
    },
};
